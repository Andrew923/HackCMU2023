CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230604000000_e20230604235959_p20230605021832_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-05T02:18:32.807Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-04T00:00:00.000Z   time_coverage_end         2023-06-04T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�;B   �          A��?E�?��A��B��qB�?E��:�HA�B�G�C�1�                                    Bx�;P�  
�          A=q?��H@��@��B�BuQ�?��H��Q�A ��B��fC��                                     Bx�;_L  �          A ��?�=q@
�H@���B�33Bi��?�=q����@�ffB��C��                                    Bx�;m�  T          A{?���@Q�@�
=B���Bd��?��׾��A (�B�aHC��f                                    Bx�;|�  �          A=q?�\)@��@�B�aHBSz�?�\)��@�
=B��fC�Ф                                    Bx�;�>  T          Aff?�=q@
=@��RB�8RBTff?�=q�#�
@��B�u�C��                                    Bx�;��  T          A�H?���@G�@�Q�B��BP\)?��þ��A Q�B�� C�8R                                    Bx�;��  T          A�?�\)@�@���B�#�BR�?�\)�.{A�B�8RC�                                    Bx�;�0  
�          A\)?�33@��@�
=B�ffBn�H?�33=�AG�B�B�@���                                    Bx�;��  �          A��?��H@�@�33B���BG�?��H��=qAB��HC�xR                                    Bx�;�|  T          A�R?�ff?�(�@��\B���B?�ff�L��@�B�{C��R                                    Bx�;�"  T          Az�?���?�z�@�ffB���B7��?��Ϳ&ffAp�B��
C���                                    Bx�;��  �          A�R?��
?��H@��HB�L�B@�
?��
�\)A   B��C��f                                    Bx�< n  "          Aff?У�?�
=@��B���B7�H?У׿z�@��RB��C�(�                                    Bx�<  �          Aff?�?�p�@�{B�\)B<Q�?�����A�HB�33C��=                                    Bx�<�  T          A\)?��H?�A�B�k�B<�R?��H�   A(�B��3C���                                    Bx�<,`  T          Aff?��
?�=qA��B��{B7��?��
�@  A�B��\C��=                                    Bx�<;  �          A��?˅?�
=A (�B��B'�H?˅�^�RAp�B��RC���                                    Bx�<I�  "          Az�?�z�?��H@�\)B�B$��?�z�Tz�A ��B�C��                                    Bx�<XR  
�          A?��
?�(�A ��B��B�R?��
�Tz�A{B��)C�t{                                    Bx�<f�  "          A�R?�p�?�G�A��B�L�B$
=?�p��O\)A
=B��C�o\                                    Bx�<u�  
Z          A=q?�z�?��HAp�B�� B%�\?�z�\(�A�HB�Q�C�P�                                    Bx�<�D  T          A�?޸R?���@��B��\B?޸R�aG�@�\)B��C���                                    Bx�<��  
�          A�R?�?�p�@���B�B7p�?���@�\)B�8RC�n                                    Bx�<��  T          A?��?�\)@��B�aHBCQ�?�녾�
=A
=B���C�˅                                    Bx�<�6  	          A�
?��H?�Q�Ap�B��\BB(�?��H�ǮA��B�(�C��                                     Bx�<��  �          A�?���@{@��B�.Bf�?���>B�\A�
B�Q�@�                                      Bx�<͂  �          A  ?��@`  @��
B=qB��?��?�  A��B�B+�\                                    Bx�<�(  	�          A��?�{@�HA ��B�33Ba?�{=���A=qB�B�@n{                                    Bx�<��  T          A	�?�
=@333@�p�B�(�Bk��?�
=?   A{B�Q�A��R                                    Bx�<�t  "          A��?�z�@7�@�p�B�  Bff?�z�?�A�\B�u�A�
=                                    Bx�=  �          A	?��
@8Q�A (�B�  B�
=?��
?\)A�
B�G�A�G�                                    Bx�=�  "          A
ff?�ff@)��A{B��B�?�ff>��RA��B�G�AT��                                    Bx�=%f  
�          A�
?��\@{A��B�B{=q?��\=�Q�A
ffB�L�@���                                    Bx�=4  �          A	��?���?�AB��Bb�?��Ϳ0��A�
B��\C��                                    Bx�=B�  �          A
ff?��\@'
=AffB�ǮB�Q�?��\>�\)A��B�ǮAD��                                    Bx�=QX  T          A(�?���@6ff@���B�k�B�=q?���?z�A�B�{A���                                    Bx�=_�  
Z          A��?�ff@1�@�
=B��B��
?�ff>��HA�HB�L�A���                                    Bx�=n�  �          A
=?��@2�\A�B�aHB��
?��>��A	�B��{A��                                    Bx�=}J  �          A	�?��\@:=qA (�B���B��q?��\?(�A  B�.A�ff                                    Bx�=��  "          A\)?��H@z�A��B�Bz33?��H��A	�B�\C�E                                    Bx�=��  "          A�?�Q�@G�AffB��HBn(�?�Q쾳33A
{B��3C��3                                    Bx�=�<  �          A
�H?�  ?���A��B�{BL  ?�  ���HA��B�
=C��                                    Bx�=��  "          A\)?!G�@Q�A�RB�u�B��f?!G��uA
�HB��C��f                                    Bx�=ƈ  �          A	G�?n{@33AQ�B��=B�G�?n{��=qAQ�B�
=C���                                    Bx�=�.  
�          A\)@
�H�}p�A�B�.C�@
�H�I��@�G�B|\)C�N                                    Bx�=��  
�          A33?��R�ǮA33B�33C�o\?��R�&ff@�G�B�k�C��                                     Bx�=�z  
�          A�
?�Q�>.{A   B���@�  ?�Q��   @�  B�u�C�{                                    Bx�>   
(          Aff@z�    AG�B��C��R@z���@���B�u�C�c�                                    Bx�>�  
�          Ap�@Q�>.{A ��B�L�@��@Q�� ��@��B��fC�Y�                                    Bx�>l  
�          Aff@
�H=�G�A��B�  @<(�@
�H�z�@��HB��C�&f                                    Bx�>-  �          A�\@�>�p�AG�B��A
=@녿���@�z�B�k�C��q                                    Bx�>;�  
�          A�\@�?!G�A�B�W
Ax(�@녿Ǯ@�{B��3C��
                                    Bx�>J^  �          A��@>�(�A��B��A8Q�@��A��B�L�C��\                                    Bx�>Y  "          A
=@�H?B�\A ��B���A�p�@�H��@��RB��C���                                    Bx�>g�  
Z          A\)@33?aG�Ap�B�.A�33@33���A z�B�� C�)                                    Bx�>vP  �          Aff@�
?c�
A z�B���A��\@�
���
@�
=B�L�C���                                    Bx�>��  
�          A
=@�R?���A ��B��A�z�@�R��\)A ��B��{C��                                    Bx�>��  "          A�@(�?��HAB��RA�33@(��}p�A=qB��C��3                                    Bx�>�B  �          A	G�@
=?�\)A�
B���A�{@
=����A�
B��qC�33                                    Bx�>��  �          A�@ff?�{A�B�aHA��
@ff���A{B�� C�XR                                    Bx�>��  �          A�R?�(�>���A ��B�#�Ap(�?�(�����@��B��)C���                                    Bx�>�4  E          A�?��H>��RA33B��3Ab�\?��H��\)@��B���C�o\                                    Bx�>��  "          A�?�=q?E�A�
B���A���?�=q��
=AffB�.C��                                    Bx�>�  �          A
=?�?B�\A��B���A�ff?�����A33B�=qC�>�                                    Bx�>�&  �          Ap�?��?�{A ��B���A�?�׿�ffA ��B�
=C�s3                                    Bx�?�  
�          A��?���?�ffAp�B��fA�z�?��ÿ�{AG�B�z�C�O\                                    Bx�?r  �          A��?�p�?��
A ��B�A�Q�?�p���{A ��B�u�C�L�                                    Bx�?&  
�          Aff?�?���A=qB�A�(�?�����A{B���C�aH                                    Bx�?4�  �          Aff?�
=?���A��B��A�?�
=���
AB�ǮC�H                                    Bx�?Cd  �          A�
@�?�AffB���A�R@녿}p�A�HB�(�C��\                                    Bx�?R
  �          A��@ ��?�33A�HB���B�R@ �׿G�AQ�B���C�n                                    Bx�?`�  "          A	p�@z�?���A�B��=B�
@z�Tz�A��B��C��                                    Bx�?oV  
�          A	p�@�\?�z�A�B�k�B
�@�\�E�A��B���C���                                    Bx�?}�  �          A��@�?=p�A ��B���A��@녿�\)@�
=B��C�                                      Bx�?��  
�          A�
@   ?O\)@��RB��A�z�@   ���
@���B��{C��
                                    Bx�?�H  
�          A
=@ff?E�@�(�B�aHA���@ff���@��B�
=C�/\                                    Bx�?��            @�ff?�  ?B�\@�  B���A�\)?�  ����@�ffB�\C��H                                    Bx�?��  "          @���?8Q�?�@��B��B{��?8Q���@�G�B�u�C�n                                    Bx�?�:  T          @���
=q@��@��BG�B�Ǯ�
=q@�@�(�B=��B�                                    Bx�?��  "          @�{�l(�@�G�?�Q�A<��B�\�l(�@�\)@7�A�G�B�z�                                    Bx�?�  T          @������\@�z῁G���
=B��3���\@�ff?�@���B��                                    Bx�?�,  �          @�p���z�@����W�C���z�@ƸR��=q��C @                                     Bx�@�  �          @���Q�@����-p����C����Q�@�G����\��C=q                                    Bx�@x  
�          @��
���@�Q��(���ffC�{���@�z�5��  C�3                                    Bx�@  T          @�����@�33�ff�v�RC�����@�{�
=q�|(�C �                                    Bx�@-�  
�          @�z���33@��������33C�H��33@��R�h����(�C�                                    Bx�@<j  T          @�33��@���)����{C����@�����	Cz�                                    Bx�@K  �          @�\)��z�@�G��\(���{C�R��z�@�����R�f�HC8R                                    Bx�@Y�  �          @��
��Q�@��R�^�R��(�Cp���Q�@�p���
�rffC ��                                    Bx�@h\  T          @�����=q@��R�x����C����=q@�G��$z�����C�                                    Bx�@w  �          @�����H@����y������C
u����H@�(��,������CT{                                    Bx�@��  T          @����  @�  �^{��G�Cu���  @�
=�
=q��  Cu�                                    Bx�@�N  c          @�z���  @�G��U�י�C.��  @�\)����(�Cc�                                    Bx�@��  �          @�\)����@��\�I����33C������@��R�����dz�C33                                    Bx�@��  �          @����H@���tz���{C&f���H@��������{C�                                    Bx�@�@  
�          @�Q���G�@��׿Y������C ���G�@�G�?.{@���C k�                                    Bx�@��  "          @�  ��z�@��R>�\)@��B��3��z�@�ff?�  AZffB��                                    Bx�@݌  1          @�G���p�@�33�}p���CL���p�@�p�>�(�@Q�C�                                    Bx�@�2  
�          @�\���R@���5���C=q���R@��?5@�(�C=q                                    Bx�@��  
i          @�\��G�@�  >�ff@]p�B�33��G�@�ff?�AlQ�C ��                                    Bx�A	~  
�          @���@��
>�ff@Y��B�z���@��@ ��At(�B��                                    Bx�A$  �          @������
@�33<�>uB�G����
@���?��
A8��C 
=                                    Bx�A&�  
�          @�  ��z�@�(�    ���
B�\��z�@�{?ǮA:=qB��R                                    Bx�A5p  �          @�����{@�z�#�
��\)B����{@�ff?��
A4��B�G�                                    Bx�AD  �          @�
=��
=@ҏ\��{��HB�k���
=@θR?��
A(�B�aH                                    Bx�AR�  �          @������@�G��W
=���B�����@���?��A%B��                                    Bx�Aab  T          @��R��=q@���B�\��\)B��f��=q@�Q�?�Q�A&�RB��                                    Bx�Ap  �          @������\@�G������{B��f���\@Ӆ?�@u�B�W
                                    Bx�A~�  T          @�=q��{@�G��c�
��=qB�aH��{@��?8Q�@�Q�B�33                                    Bx�A�T  �          @�������@Ӆ�&ff���B�Ǯ����@�=q?u@��B�{                                    Bx�A��  
�          @��\�s33@�=q�(���Q�B�33�s33@أ�?�ff@�{B�{                                    Bx�A��  E          @����~{@����G��N{B���~{@�=q?�
=A�B�W
                                    Bx�A�F  �          @��R�?\)@�=q>L��?�  B��)�?\)@�=q?��AdQ�B�ff                                    Bx�A��  �          @�z��:�H@��>��?���B���:�H@�=q?�=qA^=qB�W
                                    Bx�A֒  �          @����/\)@��>8Q�?��B�\)�/\)@��?�Ad(�B޽q                                    Bx�A�8  T          @���,(�@�\>.{?�(�B�L��,(�@�\?�z�AbffBݙ�                                    Bx�A��  �          @�
=�/\)@�
=>��R@p�B�Q��/\)@�@�AqB���                                    Bx�B�  "          @�ff�<(�@��
>�p�@*�HBߊ=�<(�@�=q@
=AvffB�33                                    Bx�B*  T          @��<(�@��H>�@U�Bߙ��<(�@��@(�A�(�B�aH                                    Bx�B�  
�          A ���7�@�\)?+�@�Q�B�  �7�@�@�HA�  B���                                    Bx�B.v  	�          @�\)�L(�@���?
=@�
=B�\)�L(�@�{@�\A�\)B�p�                                    Bx�B=  1          @�
=�G�@���?:�H@�G�B�aH�G�@��@�A�{B䞸                                    Bx�BK�  w          @��R�AG�@陚?�G�@��B����AG�@ۅ@,(�A�(�B�\                                    Bx�BZh  �          @���@��@�z�?fff@�z�B���@��@�\)@#33A���B��                                    Bx�Bi  
�          @���AG�@�p�?���@��B���AG�@�
=@.{A�{B�W
                                    Bx�Bw�  w          @��R�L(�@�Q�?#�
@�(�B�k��L(�@��@z�A��B�=                                    Bx�B�Z  "          @��R�S�
@�ff>��?���B��H�S�
@�ff?�\)A`��B�=                                    Bx�B�   T          @�p��Q�@��?z�@��\B��)�Q�@��H@
�HA�33B�                                    Bx�B��  �          @���@  @�p�?��RA��B��)�@  @�ff@333A�(�B��f                                    Bx�B�L  T          @��8Q�@��?�\)A	p�B��f�8Q�@��
@*=qA���B�R                                    Bx�B��  T          @���9��@أ�?�G�A8��B�Q��9��@Ǯ@A�A��B�Ǯ                                    Bx�BϘ  �          @�G��AG�@�
=?��A=G�B�k��AG�@�{@C33A���B�\                                    Bx�B�>  T          @�  �AG�@�?�{AE�B��AG�@�(�@FffA���B�p�                                    Bx�B��  T          @����G�@�33?��
AZ�HB���G�@���@O\)A�=qB��)                                    Bx�B��  �          @��H�G�@�z�?�z�Ai��B�B��G�@���@XQ�AӅB��                                    Bx�C
0  �          @�{�b�\@Ӆ?��HAMB�Q��b�\@�G�@J�HA���B��                                    Bx�C�  T          A�H��  @�z�?�ffAK�B�B���  @ə�@S�
A���B�Ǯ                                    Bx�C'|  "          AG���G�@��@Q�AuG�B���G�@�  @eA���B��                                    Bx�C6"  T          A�����@�{@ ��A��B������@�ff@y��A�\)B�#�                                    Bx�CD�  
�          A��~�R@�  @�RA��RB�
=�~�R@���@|(�A�33B��                                    Bx�CSn  E          A��g
=@�33@R�\A�G�B�W
�g
=@�{@�{Bp�B�                                    Bx�Cb  "          A�
�\(�@�(�@^{A�G�B��H�\(�@�@��
B�HB�ff                                    Bx�Cp�  "          A�\���@Ӆ@��At��B�p����@��R@c�
AЏ\B��)                                    Bx�C`  T          A��\)@Ӆ@ ��A�{B�33�\)@�z�@z�HA�
=B�B�                                    Bx�C�  �          A   �C�
@��@W�A�B���C�
@���@�  B�
B��
                                    Bx�C��  "          @�{�>{@θR@`��A�ffB�u��>{@���@��B�B��                                    Bx�C�R  
Z          A33�j�H@Ӆ@HQ�A��B����j�H@�Q�@���B��B���                                    Bx�C��  
�          A�
���\@׮@I��A�z�B�aH���\@�(�@��B{B��                                    Bx�CȞ  T          A33���\@�p�@N�RA�p�B��)���\@���@��
BQ�B�=q                                    Bx�C�D  T          A�H�w
=@Ӆ@9��A��B�{�w
=@��@���A�{B�.                                    Bx�C��  T          A(��tz�@�\)@8��A��
B�.�tz�@�{@�G�A�z�B��\                                    Bx�C��  
�          A	��|(�@�Q�@Dz�A�
=B���|(�@�p�@���A��B�\                                    Bx�D6  "          A33����@���@I��A���B������@�p�@�33A�\)B��\                                    Bx�D�  "          A
�H���H@��H@7�A�ffB�  ���H@ə�@��HA�  B�
=                                    Bx�D �  �          A	����ff@��
@�A�  B�
=��ff@�{@vffA�
=B�33                                    Bx�D/(  �          A	p�����@���@{A�  B�����@�ff@}p�AۮB�8R                                    Bx�D=�  "          A	��xQ�@�(�@1G�A��RB���xQ�@��
@��A���B�3                                    Bx�DLt  "          AQ��j�H@�=q@>{A��B��H�j�H@ȣ�@�p�A��\B�R                                    Bx�D[  T          A	�Z=q@���@uA�(�B虚�Z=q@�p�@�\)B�RB��f                                    Bx�Di�  �          A
ff�S�
@�
=@y��A�z�B����S�
@�\)@���B�B��                                    Bx�Dxf  
�          A���Y��@��@^{A�\)B�ff�Y��@��@���B\)B�Ǯ                                    Bx�D�  w          @�z��w
=@��[�����B���w
=@����ff�w33B�8R                                    Bx�D��  �          @�33��p�@����6ff��{B�p���p�@�\)��G��1p�B�z�                                    Bx�D�X  "          @�z���=q@�{�G����
B��q��=q@��ÿh����z�B��f                                    Bx�D��  �          @�����
=@�=q��
=�e�B�Ǯ��
=@��H�(����RB�k�                                    Bx�D��  T          @��H���H@\���H�K
=B�����H@ə������<��B��                                    Bx�D�J  �          @�(���  @ȣ�����|Q�B�8R��  @ҏ\�G���{B��                                    Bx�D��  T          @�
=����@θR�����
B��f����@��>k�?�33B��                                    Bx�D�  
�          A ����@θR��\)��\)B�����@љ�>�  ?�\B�\                                    Bx�D�<  c          A Q����@���������B������@У�>�z�@z�B��                                    Bx�E
�  w          @����Q�@У׿u���B�Q���Q�@ҏ\>��@:�HB��
                                    Bx�E�  �          @�����
@�33��  �,��B�����
@У׾#�
���B���                                    Bx�E(.  T          @�ff��G�@��H����P(�C (���G�@ʏ\�   �g
=B�33                                    Bx�E6�  
�          @�p���
=@�녿��`Q�B�����
=@�=q�#�
���
B�p�                                    Bx�EEz  "          @������
@�Q�У��A�B�\���
@�
=��p��-p�B�33                                    Bx�ET   
�          @����mp�@��
>�p�@0��B�z��mp�@��?��HAO�B�{                                    Bx�Eb�  T          @��R��  @�  ?�@\)B�.��  @�Q�?�A^�RB�#�                                    Bx�Eql  �          @��R��G�@˅?��
@���B�\��G�@���@33A���B�\                                    Bx�E�  T          A�����@���@_\)A�(�Cu�����@��\@��B�HC�                                    Bx�E��  
�          Aff���@�  @X��AĸRC���@�ff@�
=B\)C@                                     Bx�E�^  �          AG���z�@�33@G
=A���C u���z�@��@�
=A�\)CG�                                    Bx�E�  	�          Ap����@��@�As�
B��q���@�Q�@UAÅC c�                                    Bx�E��  �          Ap���  @Ǯ?���A_33B��{��  @�\)@I��A���C�                                    Bx�E�P  �          A �����\@ȣ�?��
A/33B�8R���\@��H@/\)A�33C�{                                    Bx�E��  �          A Q���
=@�
=?^�R@�Q�C ����
=@�p�@�Aqp�C                                    Bx�E�  "          A (���@�G��L�;ǮC ���@�?�Q�A��C ��                                    Bx�E�B  T          A Q���=q@�
=?&ff@�{C5���=q@�\)?�\AK�Cp�                                    Bx�F�  �          AG�����@���<#�
=���C^�����@��?��A=qC�3                                    Bx�F�  T          A �����H@��R�5��33C�
���H@�\)>�@S33C�R                                    Bx�F!4  �          @��
���@���^�R�˅CǮ���@�G�>��R@�C��                                    Bx�F/�  
�          @�Q��k�@�=q=#�
>��B��k�@�?�{A!G�B���                                    Bx�F>�  
�          @�
=�o\)@׮��33�(��B��o\)@�?s33@�33B�p�                                    Bx�FM&  T          @���u@ڏ\���H�fffB�R�u@ٙ�?Tz�@�=qB��                                    Bx�F[�  �          @����\)@���aG���33B��)�\)@ҏ\?��@��\B�z�                                    Bx�Fjr  T          @�G��{�@�
=>B�\?��B��{�@��?�(�A-�B��H                                    Bx�Fy  c          @�(��n{@���#�
��ffB�=�n{@��?.{@�\)B�\                                    Bx�F��  E          @���dz�@��(����=qB�u��dz�@�?+�@��B�z�                                    Bx�F�d  
�          @���e�@љ�������HB�W
�e�@�(�>B�\?�z�B�R                                    Bx�F�
  T          @�\)�)��@θR��{�n=qB��\�)��@�{�(����B�(�                                    Bx�F��  
�          @�\)�c33@�\)�Tz��ӅB�G��c33@���>�33@3�
B��                                    Bx�F�V  T          @�G���ff@��
�
=q����B�.��ff@��
?�@�Q�B�33                                    Bx�F��  �          @������@�
=�E��\B�\����@���>��R@p�B���                                    Bx�Fߢ  T          @�\)�j=q@��
=���RB�B��j=q@�{?�@�Q�B�=q                                    Bx�F�H  �          @���dz�@˅�
=q��ffB�{�dz�@�33?&ff@��B��                                    Bx�F��  �          @�z��Vff@�녾���N{B���Vff@���?L��@�  B�\)                                    Bx�G�  
�          @�33�333@�Q��^{���B�aH�333@���\)��=qB�k�                                    Bx�G:  
�          @�R�*�H@J=q��{�a
=C&f�*�H@����Q��>�
B�L�                                    Bx�G(�  �          @�G��9��@w�����@�
B���9��@�G������B�\)                                    Bx�G7�  c          @��^�R@���
=�C .�^�R@�
=�u���B��                                     Bx�GF,  E          @��H�\��@��H��G��Q�B�\�\��@����U���B��                                    Bx�GT�  �          @�p��W
=@�ff��\)���B���W
=@�(��S�
��ffB��f                                    Bx�Gcx  �          @�z��hQ�@����vff��
B��\�hQ�@�z��;���=qB���                                    Bx�Gr  
�          @��l��@��
�hQ����
B�z��l��@�{�,�����RB�p�                                    Bx�G��  �          @�\�l��@�
=�Z�H��p�B�W
�l��@�  ��R��=qB��)                                    Bx�G�j  �          @���j�H@�33�`  ��33B�33�j�H@����%����\B�p�                                    Bx�G�  �          @�ff�r�\@�G��S�
��C T{�r�\@����=q��=qB�{                                    Bx�G��  �          @�Q��`  @��b�\����B��R�`  @�
=�'����B�#�                                    Bx�G�\  T          @߮�J=q@�  �mp�� =qB����J=q@��\�1G�����B��                                    Bx�G�  
(          @߮�2�\@��H�xQ��
=B��2�\@�{�;���ffB�L�                                    Bx�Gب  
�          @�z��8��@�
=�z�H�G�B��
�8��@��\�<����\)B�3                                    Bx�G�N  "          @��8Q�@�Q��s�
���B�p��8Q�@��H�5��=qB�                                    Bx�G��  �          @�ff�=p�@��\�tz�� G�B�\�=p�@���5��  B�.                                    Bx�H�  
�          @�(��5@����vff���B��5@���8����33B��                                    Bx�H@  �          @�\�2�\@�=q�j�H���B�ff�2�\@��
�,����ffB��f                                    Bx�H!�  T          @����)��@�  �hQ����B�z��)��@����+���z�B��                                    Bx�H0�  
(          @�=q�2�\@�z��E�י�B���2�\@��H����\)B�                                    Bx�H?2  "          @޸R��\@����b�\��(�B��
��\@���"�\��p�B�W
                                    Bx�HM�  T          @�����@�Q��aG���G�B�=q��@����!����\B��
                                    Bx�H\~  T          @ۅ�{@�(��N{��{B�\�{@��H��R����B�\                                    Bx�Hk$  
�          @�33��@����L(��߅B� ��@�  �{��  B�\)                                    Bx�Hy�  
Z          @�ff�   @���9����
=B��f�   @�z��{�w�
Bֽq                                    Bx�H�p  
�          @��H�p�@�p��Fff��  Bޅ�p�@Å����ffB۽q                                    Bx�H�  
�          @��H���@��
�K��޸RB����@���p���p�B��
                                    Bx�H��  �          @�z���
@�G��K��ݮB�(����
@Ǯ�(�����B��)                                    Bx�H�b  �          @�(���
=@�G��S�
��(�B�p���
=@�  �z���p�B�#�                                    Bx�H�  �          @�z���
@���C�
�ҏ\BՏ\���
@ʏ\�33��ffB�u�                                    Bx�HѮ  
�          @ڏ\��ff@�
=�-p�����BՏ\��ff@ʏ\�ٙ��g�Bӽq                                    Bx�H�T  �          @أ����@��0����\)B�3���@������x��B�(�                                    Bx�H��  T          @�ff�@  @�Q��R�\��p�B�Ǯ�@  @�
=�(���
=B�ff                                    Bx�H��  �          @ٙ��R�\@��R�hQ��{B��H�R�\@�\)�5��ď\B�W
                                    Bx�IF  �          @׮�7
=@�Q��\(�����B�u��7
=@�  �&ff��  B�                                      Bx�I�  �          @��B�\@��_\)�{B��f�B�\@��/\)�ʸRB�Q�                                    Bx�I)�  
�          @Ϯ�C33@��
�W�����B���C33@��H�&ff���HB��)                                    Bx�I88  �          @���<��@����U����RB���<��@����!���{B�                                     Bx�IF�  "          @��H�)��@�33�N{���
B����)��@����Q���33B�(�                                    Bx�IU�  �          @�
=��@����Q��陚B�����@�33������HBޞ�                                    Bx�Id*  �          @��p�@����W���RB��p�@���{����B�q                                    Bx�Ir�  �          @�\)���@�G��Fff�ҏ\B�����@�ff�
=q��ffB�8R                                    Bx�I�v  �          @���  @���8Q����B���  @Ǯ��Q�����Bۣ�                                    Bx�I�  �          @�Q��	��@�33�)����(�B�Q��	��@�{��(��k\)B�G�                                    Bx�I��  �          @Ӆ����@�=q������B�����@�(����
�V=qB�G�                                    Bx�I�h  �          @����
=@����,(����
B�k���
=@�z�޸R�j{Bգ�                                    Bx�I�  �          @׮��@�{�1G���p�B�Q���@�G��������B�.                                    Bx�Iʴ  �          @����Q�?(���}p��
=C,�
��Q�?�  �s�
��
C&��                                    Bx�I�Z  �          @ָR��G�?&ff���\��
C,�R��G�?�G��|(���RC&�H                                    Bx�I�   �          @�G����?5��z��!�C,{���?�{�����HC%)                                    Bx�I��  �          @ٙ���33?c�
�p  ���C*����33?�Q��dz���C%��                                    Bx�JL  
�          @�G���\)?z�H�xQ��z�C)�)��\)?���l(���
C$G�                                    Bx�J�  �          @أ�����?�z��~{��C%
����?�p��n{�\)C��                                    Bx�J"�  �          @����{?�G������ffC!T{��{@��n�R�{C޸                                    Bx�J1>  T          @�G���@z��w
=�
��C33��@'
=�aG���  C=q                                    Bx�J?�  "          @��H���
@#�
�n{�=qCu����
@Dz��S�
��Q�C�                                    Bx�JN�  �          @��H���@,(��]p���C�)���@J=q�B�\��p�C�                                    Bx�J]0  �          @�G���ff@:=q�P  ����C�=��ff@U�333����CB�                                    Bx�Jk�  �          @����=q@Vff�AG���p�C����=q@p  � ����C�=                                    Bx�Jz|  �          @��,��@�
=� �����B�u��,��@�ff���H�-G�B枸                                    Bx�J�"  T          @�{�#�
@��\��(���{B�G��#�
@�����z��%��B㙚                                    Bx�J��  T          @����(��@�(���R���B�G��(��@�zΌ���Q��B�.                                    Bx�J�n  T          @�(��+�@�p��%��  B��)�+�@�\)����z�B�(�                                    Bx�J�  �          @�33�*=q@�G��33��{B�ff�*=q@�녿���a�B�(�                                    Bx�Jú  �          @���2�\@��#33��G�B왚�2�\@�\)��ff����B��                                    Bx�J�`  �          @�=q� ��@�p��%��ffB�R� ��@�\)��{���B�(�                                    Bx�J�  �          @�  �!G�@��H�/\)����B� �!G�@�p���p����B���                                    Bx�J�  �          @���#33@�  �)����33B��
�#33@�녿�z���z�B�G�                                    Bx�J�R  �          @���%@�G���\���B�8R�%@�����ff�b�HB�{                                    Bx�K�  �          @�
=�Q�@�G������33B� �Q�@�=q�У��h  B��=                                    Bx�K�  �          @���
=@�33�{����B�q�
=@�(��ٙ��o�
B�                                    Bx�K*D  �          @θR�(�@�Q��!G���ffB�u��(�@�G���G��~ffB�z�                                    Bx�K8�  �          @�  ��@��\�&ff���B���@��
����  B�(�                                    Bx�KG�  T          @�
=���H@�G��*=q���B�\���H@��H��z���B�.                                    Bx�KV6  �          @�
=�z�@�Q��+����
B�(��z�@�녿�
=���B�(�                                    Bx�Kd�  �          @�녿�Q�@�\)�,(����
B��ÿ�Q�@��ÿ���(�B�aH                                    Bx�Ks�  �          @�{�G
=@���Mp����B�\�G
=@���%���
B��                                    Bx�K�(  �          @˅��
=@W��<����=qC  ��
=@n{�\)��\)C:�                                    Bx�K��  �          @�{��  @9���5��У�C���  @O\)����C
                                    Bx�K�t  �          @���  @@  �Fff����C����  @W��,(���{C�H                                    Bx�K�  �          @�(����
@'��H����z�C�R���
@@  �1���G�Cc�                                    Bx�K��  T          @�z����@   �>�R����C�R���@6ff�(���îC��                                    Bx�K�f  
�          @ʏ\���H@3�
�<����Q�C�
���H@J=q�$z���  C�)                                    Bx�K�  
�          @˅�\��@�z��Tz����C���\��@����1G���=qB��3                                    Bx�K�  T          @�Q��`  @���P  ��{C p��`  @����+��¸RB�k�                                    Bx�K�X  T          @�Q��<(�@�=q�Mp����B��H�<(�@�p��%��Q�B�B�                                    Bx�L�  
�          @��K�@�33�Fff��Q�B�B��K�@�ff�!G���p�B�p�                                    Bx�L�  T          @Ǯ�L(�@�33�I�����
B�u��L(�@��R�%��  B�B�                                    Bx�L#J  
�          @�Q��C�
@���U����HB���C�
@����/\)��B�{                                    Bx�L1�  "          @�(��5@�  �I����G�B�q�5@�33�#�
��G�B�B�                                    Bx�L@�  �          @�=q�0��@��R�G
=��p�B�3�0��@����!����B�G�                                    Bx�LO<  
Z          @�  �>{@�ff�s33�	G�B����>{@��
�Mp���B�B�                                    Bx�L]�  T          @�Q��G
=@�  �hQ��=qB�u��G
=@����B�\��z�B�Q�                                    Bx�Ll�  
�          @�33�8��@�{�n�R�ffB��8��@�33�HQ��ٮB�Ǯ                                    Bx�L{.  
�          @�33�.�R@�33�tz���B���.�R@�Q��J�H���B�Ǯ                                    Bx�L��  �          @��H�.{@�\)�z�H�\)B��f�.{@����Q�����B�\)                                    Bx�L�z  �          @��0  @��\�z=q�z�BꞸ�0  @���QG���p�B�33                                    Bx�L�   �          @�ff����@�=q�l��� �B�Ǯ����@�ff�B�\�ϮB֊=                                    Bx�L��  �          @��ÿ�=q@�  �n{�z�B����=q@�(��Dz����B�ff                                    Bx�L�l  �          @ڏ\��  @�33�l(����B�=q��  @�\)�A���
=Bˮ                                    Bx�L�  T          @ڏ\���@�
=�|(��{B��ῇ�@�(��S33��G�B�z�                                    Bx�L�  T          @���k�@�(���\)�33B�\)�k�@�=q�fff��B��                                    Bx�L�^  �          @�{�Y��@��R�����$�
B�k��Y��@���tz����B���                                    Bx�L�  �          @�zῂ�\@�����ff�$33B��쿂�\@�G����H�=qB�                                    Bx�M�  T          @��ÿ���@�Q���=q�{B̔{����@�\)�|(��ffBʳ3                                    Bx�MP  �          @��
��ff@��H��\)���B�LͿ�ff@�Q��fff��  B���                                    Bx�M*�  "          @�=q�
=@�ff��G���\B�aH�
=@�33�Z�H��B��                                    Bx�M9�  "          @�  ����@��\���
�  B�����@����s33��
B��                                    Bx�MHB  �          @���
=@�33��  �  B����
=@����y���{B��                                    Bx�MV�  �          @�p���@�=q����B�Ǯ��@�Q��tz�� B���                                    Bx�Me�  �          @陚���@����(��p�BԀ ���@�G��qG���=qB�L�                                    Bx�Mt4  T          @����z�@�  ��Q��B�G��z�@�p��j=q��ffBڙ�                                    Bx�M��  �          @�=q���@��h����(�B�p����@����>�R���B�aH                                    Bx�M��  �          @���
=@��H�s�
��\B�(��
=@θR�J=q�ǮB�(�                                    Bx�M�&  �          @��ff@\�vff��\)B�#��ff@�{�Mp��ʸRB��                                    Bx�M��  �          @��R�(�@�\)�XQ���ffB����(�@�G��+���Q�B�.                                    Bx�M�r  T          @�\)�
�H@�\)�z����B׮�
�H@��У��J�\BָR                                    Bx�M�  �          @��H���@�z����yG�B�u����@�녿�=q�"{Bٙ�                                    Bx�Mھ  �          @�
=�@�G������#33B���@�(��(����RBԮ                                    Bx�M�d  �          A����@�G���(��B�8R���@�{�n{��ffB�33                                    Bx�M�
  �          A  �@�33�j=q���HBӮ�@��:�H��p�B�k�                                    Bx�N�  T          A�׿�33@�G���G���
=B�uÿ�33@����S�
��\)B�Q�                                    Bx�NV  T          A(�����@�����(����B�
=����@�=q�|(����
B̙�                                    Bx�N#�  �          A  ����@��H���H��RB̀ ����@����z���Q�B�{                                    Bx�N2�  �          A33��(�@�������(Q�B�33��(�@�����(�B��)                                    Bx�NAH  �          A	��<�@�{����B��\<�@�p���ff�
=B���                                    Bx�NO�  �          A{���@�ff��(���B��
���@�������33B�ff                                    Bx�N^�  �          A���Q�@����H�
��B�\��Q�@�z���
=��B�ff                                    Bx�Nm:  �          A���i��@ٙ������	ffB�ff�i��@�Q����
��
=B�W
                                    Bx�N{�  �          A����@��H����)B�aH���@�=q�����G�B�                                    Bx�N��  
�          A���"�\@��
��\)�*�
B�{�"�\@�33��
=���B��                                    Bx�N�,  
�          @��H�*=q@��u��
B�(��*=q@�Q��Y����\)B��f                                    Bx�N��  �          @ָR�0  @��n{��B�Q��0  @���QG���33B�W
                                    Bx�N�x  T          @�ff�+�@�����
=�*  B�.�+�@�
=��  ��
B�=q                                    Bx�N�  �          A{�@��@�  ���
�F��B��=�@��@�����ff�5z�B��                                    Bx�N��  T          @��R�33�W
=��\).C9�\�33>�(���
=��C(J=                                    Bx�N�j  �          @�z�p��@mp��ڏ\�r��B�uÿp��@�G���\)�_��B̽q                                    Bx�N�  �          @�=q��=q@X�������pB�{��=q@|(���
=�^��B���                                    Bx�N��  �          @��
��ff@N{��{�y  B�=q��ff@p�������f=qB�Q�                                    Bx�O\  �          @��
�Q�@.{�߮B�W
�Q�@S�
�׮�|BϸR                                    Bx�O  �          @�\���@Z�H��(��v��B֊=���@}p���=q�d�B�                                    Bx�O+�  �          @�(��h��@4z���G�W
B��Ϳh��@Y�������zQ�B�                                    Bx�O:N  �          @�ff��@�����z��A�HB����@�ff��\)�.�B��
                                    Bx�OH�  �          @��L��@   �߮��B�� �L��@Dz���Q�33B�
=                                    Bx�OW�  �          @�\�#�
?�z���
=ffB�=q�#�
@�R��G��B��R                                    Bx�Of@  T          @����ff?�����\ffB��þ�ff@!G�����\)B�G�                                    Bx�Ot�  T          @ᙚ?Q�?��׮#�B��?Q�@��=q��B�G�                                    Bx�O��  �          @�?У�?xQ���(�\A���?У�?�������� B0{                                    Bx�O�2  T          @�ff@���>������-�R@�p�@���?Tz���33�+\)A��                                    Bx�O��  �          @��@���?�=q��G��Q�A���@���@
�H��(��  A���                                    Bx�O�~  "          @���    @c33��33�a��B��    @|������P�B�{                                    Bx�O�$  �          @�����\@g
=<#�
=uC
xR���\@fff>�\)@<��C
�\                                    Bx�O��  �          @�  ��33@@  ����g33C����33@G
=��33�E�C��                                    Bx�O�p  �          @�33�*�H@�\)?�p�Al��B�(��*�H@��\@
�HA���B�8R                                    Bx�O�  �          @�(���Q�@�  ���R�*�RB�����Q�@�=q�Y����G�B���                                    Bx�O��  
�          @�����ff@�=q��\)�+\)C
���ff@��Ϳ��\����C
                                      Bx�Pb  D          @�ff�33@\)@(��A��
B�G��33@r�\@:�HB�B�q                                    Bx�P  
�          @��H�.{?�p�@��
B�Q�B�.�.{?xQ�@�RB��B��                                    Bx�P$�  �          @�{�L��@�\)@�  BPG�B�
=�L��@���@���B`��B��\                                    Bx�P3T  �          @�G�?˅@'�@�{B~��Bj�?˅@�@˅B�
=BWQ�                                    Bx�PA�  �          @�?�z�?���@�{B���BS
=?�z�?���@��B���B1�                                    Bx�PP�  �          @�\)?��?�Q�@���B�.B6�H?��?��R@У�B�� B��                                    Bx�P_F  �          @�G�@�@�R@��
Bq��B6(�@�@�
@���B|\)B �                                    Bx�Pm�  �          @�z�?�
=@%@ǮBz��BU�?�
=@
=q@���B�W
BA(�                                    Bx�P|�  �          @�?��\@HQ�@У�B|�RB��
?��\@,(�@ָRB���B�W
                                    Bx�P�8  �          @�p�>��H@R�\@�G�Bt�RB��>��H@8Q�@�  B��fB�                                    Bx�P��  �          @���=���@*=q@�B��{B��H=���@G�@��HB��B�(�                                    Bx�P��  �          @љ�����@Y��@�=qBj�B�\)����@A�@���Byz�B�Ǯ                                    Bx�P�*  T          @�\)>�\)@aG�@�p�Bcz�B���>�\)@J�H@�(�Br=qB���                                    Bx�P��  �          @�
=?��@n{@���BK�B�W
?��@Z=q@��BY�B��                                    Bx�P�v  �          @���?c�
@�ff@��B/��B�ff?c�
@��@�(�B=��B��f                                    Bx�P�  �          @��
>���@o\)@�(�B\�\B��3>���@X��@�33Bj�
B�                                    Bx�P��  �          @�?!G�@�(�@�(�BKG�B�Ǯ?!G�@s�
@�(�BYG�B�W
                                    Bx�Q h  �          @����?\)���?uA��RC6Ǯ�?\)�L��?s33A�
=C7�f                                    Bx�Q  T          @��
�����tz῅��(��C_T{�����p  ���\�N�RC^�\                                    Bx�Q�  T          @Ϯ�Z=q��������\�CFY��Z=q�Q���
=�`z�CA��                                    Bx�Q,Z  �          @�
=�L��>L����G�­��C)�L��?   ����§�
B���                                    Bx�Q;   
�          @�z�\�\(����R¡G�Cuٚ�\����  §�Ci�3                                    Bx�QI�  �          @ᙚ�8Q���������((�Cl�=�8Q������\)�3�Ck�                                    Bx�QXL  T          @�=q@xQ�fff�,(��(�C�|)@xQ�=p��/\)�
�
C��{                                    Bx�Qf�  T          @�G�@=q��=q������C��3@=q��{�(���z�C��3                                    Bx�Qu�  T          @������z��w
=�%��C{����z�H���H�1�Cz
=                                    Bx�Q�>  �          @�(���\)�l(����R�B\)Cy�׿�\)�\(�����N(�CxJ=                                    Bx�Q��  
�          @�  ��Q��U������U�RCzk���Q��Dz���{�affCx�H                                    Bx�Q��  �          @��0�׿�ff���\�d(�CU\�0�׿\��p��j�CP�)                                    Bx�Q�0  
�          @��b�\�9�������[�HC[Q��b�\�#33��p��cG�CW                                    Bx�Q��  T          A����H�W
=��Q��KCY����H�@  ���R�HCV��                                    Bx�Q�|  
�          Az����n{����C��C]�����XQ���33�KQ�CZ��                                    Bx�Q�"  T          A	��P  ���H��Q��<{Cl
�P  ������  �E�Cj8R                                    Bx�Q��  �          A	�2�\������p��/{Cs8R�2�\��
=��ff�9Q�Cq��                                    Bx�Q�n  �          A33�C�
��z�������Csu��C�
���
���H� ��Crk�                                    Bx�R  
�          AG��s�
������R�=qCoB��s�
��p������
�HCnL�                                    Bx�R�  �          A
{�����������R��Ce�f������  ��ff�&�HCd
                                    Bx�R%`  "          A\)����������  �  C\Y�����������ff� ��CZ�)                                    Bx�R4  "          A�R���
�����^{��
=C[޸���
��\)�l(�����CZ��                                    Bx�RB�  �          A�
�����������H���CZ���������H�����(�CY�                                     Bx�RQR  �          A����G��h����Q��ffCV���G��X����p��#33CT�                                    Bx�R_�  "          A�R��G��E���G��CQ
��G��6ff��p����CO=q                                    Bx�Rn�  
�          A���  �QG����\�33CR����  �AG���
=�$G�CP�=                                    Bx�R}D  �          A
=���\�33��G��*G�CJc����\�33��z��-��CH5�                                    Bx�R��  T          A����Q��&ff��z��+{CM=q��Q��ff��  �/(�CK)                                    Bx�R��  �          A�����׿�  �����W�CI33���׿�(���
=�Z��CF�                                    Bx�R�6  �          Ap�������33��{�6�HCEJ=������z���  �9�CB�f                                    Bx�R��  
�          A ����녿��R���H�:�CC����녿�  �����=
=CA5�                                    Bx�RƂ  T          @�\���?����\�	\)C(�q���?�=q�����ffC'5�                                    Bx�R�(  �          @�G����@��R��Q��1Cz����@�Q쿡G��\)C&f                                    Bx�R��  �          @�����\)@Tz��	������C�)��\)@Y���G��v�RCJ=                                    Bx�R�t  �          @����@��;���(�C�q��@���5���C�{                                    Bx�S  �          @���  ?ٙ��7
=���\C$�=��  ?��2�\���
C#�H                                    Bx�S�  
�          @�G��Ǯ��R�aG���Q�C9���Ǯ���H�b�\���
C8}q                                    Bx�Sf  �          @�33��\)�#�
�c�
��ffCL���\)��H�i�����CJ�                                    Bx�S-  �          @�\)���
�@  �l(���C:xR���
��R�mp���p�C9Q�                                    Bx�S;�  "          A(����0���l����  C9k����\)�n�R��\)C8aH                                    Bx�SJX  d          A���  ��\�^�R��=qCAG���  ��33�b�\���
C@ff                                    Bx�SX�            A���=q�Tz�:�H���
C9���=q�L�ͿB�\��=qC9��                                    Bx�Sg�  
�          A ����G���Q��\)�9�C<����G����׿�z��>ffC<G�                                    Bx�SvJ  
�          A ����zῢ�\���t��C=n��zῙ���
=q�y��C<��                                    Bx�S��            AQ���z�����Vff��  CF�=��z���\�[��ď\CE�                                     Bx�S��  �          Aff��p����)�����CCL���p������-p����CB��                                    Bx�S�<  �          @�ff��\��\�
=q��CA�
��\�ٙ��p����CA
                                    Bx�S��  �          @�(����Ϳ�\)�n{�أ�C?�q���Ϳ��Ϳz�H��(�C?Ǯ                                    Bx�S��  �          A Q��ٙ��#�
�I������CH�)�ٙ��p��N�R��Q�CG�f                                    Bx�S�.  T          A�\�����R�!���\)CF����=q�'
=����CF:�                                    Bx�S��  �          A���\�   ����UG�CF@ ��\��Ϳ��H�]G�CE�H                                    Bx�S�z  �          A   ���ff���
�NffCC�����33���UG�CCT{                                    Bx�S�   �          A�\��
�����G���C=��
���
��ff��p�C<��                                    Bx�T�  �          A������ff���J�HC=�������   �Z=qC<�                                    Bx�Tl  "          A  �녿��׽L�;�p�C=��녿��׽�Q�!G�C=��                                    Bx�T&  
�          A�
�=q���>�  ?޸RC;�3�=q���>aG�?��C;�q                                    Bx�T4�  T          Az��
=��G�?
=@��C;��
=���\?\)@x��C;�                                    Bx�TC^  T          @�\)��(��@  ?z�H@���C9u���(��G�?u@���C9�H                                    Bx�TR  �          @�ff���H�u?}p�@�ffC5�����H���?}p�@���C6                                      Bx�T`�  �          @��ᙚ?�\)?�(�AU�C'{�ᙚ?�=q?�  AZ�\C'c�                                    Bx�ToP  T          @�=q�أ�?Ǯ?���AK\)C'  �أ�?��
?���AP  C'J=                                    Bx�T}�  T          @�33�ᙚ��z�>�ff@c�
CA@ �ᙚ��>��@QG�CAQ�                                    Bx�T��  �          A���
=��
=�6ff��{CV&f��
=����<(�����CU                                    Bx�T�B  �          AQ����
�?\)�u��CIs3���
�>{���\���
CIQ�                                    Bx�T��  �          @�����z��q��fff��G�CW{��z��mp��j�H��\CV�\                                    Bx�T��  �          @�Q����
�P  �}p��{CS.���
�J�H�����ffCR�\                                    Bx�T�4  T          @�p�����(��C33��Q�CF+�����Q��E����CE�R                                    Bx�T��  �          @��������G��mp���z�CG�H��������p  ��\)CGT{                                    Bx�T�  �          @��R�Ǯ�\)�|(�����CGǮ�Ǯ�
�H�~�R��Q�CG33                                    Bx�T�&  �          @����Å�B�\�Z=q��Q�CNu��Å�>�R�]p����
CN�                                    Bx�U�  �          @�z����H�{��W
=��G�CW\���H�w��Z�H�ՙ�CV��                                    Bx�Ur  "          @������R�xQ��[���  CU����R�tz��_\)��  CT��                                    Bx�U  �          @�����R�N�R��  �CQxR���R�J=q������CP�R                                    Bx�U-�  "          A (���
=�O\)����
��CQ����
=�K�������CQ�                                    Bx�U<d  T          AG���ff�U�y�����
CPQ���ff�Q��|�����HCO�                                    Bx�UK
  �          @������i���n�R��
=CS������fff�q���Q�CS�                                    Bx�UY�  
�          @����{�P  ��z���CR�
��{�K����p�CR\)                                    Bx�UhV  
�          A���  �u���R�Q�CV����  �q���Q��  CV��                                    Bx�Uv�  �          A=q��\)����j=q�֏\CY�)��\)���
�mp����CYT{                                    Bx�U��  �          A{�����H�5���\)CY� ������8Q����\CY�\                                    Bx�U�H  
�          Ap���  ��z��a���G�C\!H��  ��33�e��z�C[��                                    Bx�U��  "          @�=q��{����1G����\C_  ��{���\�4z����
C^��                                    Bx�U��  T          @�R��{��Q��  ���\CX�\��{�����\��33CX��                                    Bx�U�:  �          @����H�j�H�����|  CT&f���H�j=q��p���=qCT�                                    Bx�U��  T          @��\>��?���A
=C/�\�\>�?���A�C/��                                    Bx�U݆  T          @�Q���
=?���@@��Aԣ�C"
=��
=?�=q@AG�A�C"B�                                    Bx�U�,  �          @�G����R@AG�@���B=qC�����R@?\)@��B33C(�                                    Bx�U��  �          @أ��Z�H@S�
@���B4��C���Z�H@Q�@�=qB6{C.                                    Bx�V	x  T          @�Q���H@xQ�@�\)BK��B�R���H@u@�Q�BMG�B�{                                    Bx�V  �          @љ���33@e@�  BV��Bި���33@c�
@���BX
=B�                                      Bx�V&�  �          @أ׿�  @i��@�(�BTQ�B�ff��  @g�@���BUffB�                                    Bx�V5j  �          @أ׿���@�@�(�B9{B�G�����@��@���B:�B�u�                                    Bx�VD  �          @�G���33@���@��\B3�B�k���33@�(�@�33B4
=B׏\                                    Bx�VR�  "          @��>��ÿ   ����3�
C��\>��ÿ   ����4�\C���                                    Bx�Va\  T          @���?   ��  ��ff�(�C�y�?   ��\)��
=�(�HC�|)                                    Bx�Vp  
(          A{?�33�θR��Q��,�
C�'�?�33��{�����-p�C�+�                                    Bx�V~�  �          A=q?�p���p���=q�&�C�C�?�p�������H�'  C�Ff                                    Bx�V�N  
�          Az�>�����������\C���>���ᙚ�����C���                                    Bx�V��  �          A  �#�
��=q��
=��C��R�#�
��=q��\)�(�C��R                                    Bx�V��  �          A�
����������  ��C��R����������  �=qC��R                                    Bx�V�@  �          A\)?
=q��Q�����=qC�5�?
=q��Q�����=qC�5�                                    Bx�V��  �          A��&ff���j=q�ҏ\Cyk��&ff��{�j=q��Q�Cyk�                                    Bx�V֌  "          A
=�Mp���\)�@������Cv��Mp���\)�@����ffCv)                                    Bx�V�2  �          @�Q��?\)��(��s33���Cp���?\)��(��r�\����Cp�                                    Bx�V��             @�  �&ff��ff�r�\���CrQ��&ff���R�q��33Cr\)                                    Bx�W~  �          @����R��
=�}p��ffCr@ ��R��\)�|����HCrL�                                    Bx�W$  �          A��S�
��p�������HCr���S�
��{��(��G�Cr�=                                    Bx�W�  T          A�\�g���Q���z��G�Cr�3�g����������Cr                                    Bx�W.p  
�          A��P����
=��
=��ffCu�R�P������{����CuǮ                                    Bx�W=  �          A��9����z���
=��33Cx�
�9�������{��33Cx��                                    Bx�WK�  
|          AQ��e�����G���=qCs\)�e����  ��(�Csn                                    Bx�WZb  "          A������{�{���=qCo������޸R�x����  Co�q                                    Bx�Wi  �          A���qG����8Q���p�Cr޸�qG����
�5�����Cr�                                    Bx�Ww�  �          A Q��\����׿�\�K�Cs�{�\����G���(��F=qCs޸                                    Bx�W�T  �          @�ff�P����33��z��$Q�CuO\�P������{�ffCuW
                                    Bx�W��  �          A�H��(������,(�����C�c׿�(���p��(Q���\)C�g�                                    Bx�W��  T          A�\���\�
�R�5�����C������\�33�0�����C��=                                    Bx�W�F  T          A33����=q��R{C��ῇ���\� ���J=qC��q                                    Bx�W��  �          A\)���
ff�����HC������
�\���H��RC���                                    Bx�Wϒ  
�          A33�p�����z��VffC}�)�p��녿��R�M�C}�f                                    Bx�W�8  �          A�� �������
��C���� ������Q����C��f                                    Bx�W��  T          A
�R�������G��ָRC�����n{��(�C��                                    Bx�W��  �          A(��#�
��
=��ff�G�C|:��#�
������H��C|B�                                    Bx�X
*  T          Az�����H��33���Cs3���
=�����CxR                                    Bx�X�  T          Ap����G���
=�<��C~�R���G����
��\C~��                                    Bx�X'v  �          @�Q�z�H�����
=�_\)C�3�z�H����{�T(�C�
                                    Bx�X6  �          @Å��z���
=�����q�C�C׿�z������  �e��C�H�                                    Bx�XD�  T          @�Q쿥���=q�����,��C��=����ڏ\��p�� ��C���                                    Bx�XSh  "          @�(��
=��p��!���z�C��R�
=��ff����(�C���                                    Bx�Xb  T          @�{>�������:�H��(�C�l�>�����33�5�����C�j=                                    Bx�Xp�  �          @��?u���J=q�ָRC�l�?u��\)�Dz���  C�c�                                    Bx�XZ  �          @�\)?������N�R��C���?�������I������C���                                    Bx�X�   �          @�
=?�\��(��l��� \)C���?�\��{�g
=���
C���                                    Bx�X��  �          A�R��z����R�^�R��(�C��R��z���
=�=p���
=C���                                    Bx�X�L  
�          A
=��p������G��*{C�Ǯ��p���z῰���33C���                                    Bx�X��  T          A\)��p���p��(��w
=C}q��p���ff��
�h  C��                                    Bx�XȘ  �          A{��33���  ����C��R��33���R���q��C�޸                                    Bx�X�>  T          A�H�����
����r�RC~�q�������   �b�HC~�\                                    Bx�X��  "          A\)���������qp�C+�����ff�   �aG�C=q                                    Bx�X�  �          A{����\)���qG�C|�R���������`��C}�                                    Bx�Y0  T          A���z���
=��Q��X��C}=q�z���Q��ff�H(�C}Q�                                    Bx�Y�  "          A�"�\��G��У��4(�C{��"�\��=q��p��#
=C|                                      Bx�Y |  
Z          A�׿�33��(���{�3�C�4{��33��������!C�:�                                    Bx�Y/"  T          A(��p����׿�Q��<��C~��p��������
�*�RC~.                                    Bx�Y=�  �          Az�������\�Ǯ�-C~J=��������33�33C~Y�                                    Bx�YLn  �          A(��4z���녿�G��EG�Cy�=�4z����H�����3
=Cy�H                                    Bx�Y[  "          A��G���p�����5�C��G���ff���H�!C��                                    Bx�Yi�  �          Az῾�R����E���p�C�����R�녿z��z=qC��                                    Bx�Yx`  
�          A=q��=q�33�333����C��Ϳ�=q�\)���aG�C��\                                    Bx�Y�  T          A�\��\�33�#�
���C��Ϳ�\�33<�>W
=C���                                    Bx�Y��  "          A�ÿ����R���N{C�n�����R��=q���C�n                                    Bx�Y�R  "          @��\�#�
���׽��
�z�C����#�
����=�G�?J=qC���                                    Bx�Y��  
j          A	���(��  �L�����C��3��(��Q������
C���                                    Bx�Y��  �          A33�z���׿˅�(��C���z��������HC��                                    Bx�Y�D  T          A��
=�������RC�q�
=�  ��=q���C�                                    Bx�Y��  T          A	p�� ���Q쿡G��ffC�(�� ����Ϳ�ff��\)C�/\                                    Bx�Y�  �          A(����G������ffC}G�����\��33��\)C}W
                                    Bx�Y�6  
�          A��Q��녿����
=CJ=�Q��ff�k��ȣ�CT{                                    Bx�Z
�  �          A33�33�ff�s33�ȣ�C�%�33��R�:�H����C�(�                                    Bx�Z�  �          A
�H����ff�����  C�������H�Tz����C��=                                    Bx�Z((  
�          A�H��
���
�aG��ƸRCQ���
��z�+���{C\)                                    Bx�Z6�  �          A=q��������:�H��p�C~:������=q���hQ�C~B�                                    Bx�ZEt  "          A�H�	���p��k�����C!H�	���녿0����
=C+�                                    Bx�ZT  
�          A=q�����^�R��  Cٚ���p��#�
��p�C�H                                    Bx�Zb�  
�          A���_\)��Q쾔z��p�Cu{�_\)��׽�\)��Cu�                                    Bx�Zqf            Ap���\)��\)>�=q?�
=Cl^���\)��
=>��@U�ClT{                                    Bx�Z�  !          A�
��G����?0��@�=qCp����G���(�?h��@ə�CpxR                                    Bx�Z��  "          A�H�N�R��R?
=@�z�Cv�{�N�R��{?O\)@��RCv�                                    Bx�Z�X  �          Ap��r�\��\)?�z�A ��Cq}q�r�\��?�\)A9�CqW
                                    Bx�Z��  
�          AG���������?�33A<z�Co+�������\)?���AT��Cn��                                    Bx�Z��  T          A
=q�+����H��z��P  C{��+����Ϳ��4��C{@                                     Bx�Z�J  T          A
�\�'
=��
=���H�8  C{�{�'
=� Q쿹���z�C{�                                    Bx�Z��  �          AG��xQ���=L��>�{Cs��xQ���>���?�p�Cs��                                    Bx�Z�  �          A�������  >��@ECq�������\)?8Q�@��RCq�q                                    Bx�Z�<  
�          A
{��Q����?�@^�RCnO\��Q���G�?B�\@��Cn:�                                    Bx�[�  �          A\)��G���R?���A=qCp
��G����?˅A(Q�Co�                                    Bx�[�  T          A�H�i�����@
=qAn{Cs\)�i����{@��A���Cs�                                    Bx�[!.  "          A=q�n{��z�@
=A��RCrxR�n{�ᙚ@&ffA�Q�Cr0�                                    Bx�[/�  �          A
=�e���G�@G�AyCsٚ�e���R@!G�A���Cs�
                                    Bx�[>z  
�          A��~{���@�RAs�
Cq  �~{��\@{A��Cp�R                                    Bx�[M   �          Aff�|����?��
AD(�Cq(��|����@�A`  Cp�                                    Bx�[[�  �          A
=�L����\?�33A:ffCvaH�L����Q�?�33AW�Cv33                                    Bx�[jl  "          @�p��!���?�33A#
=C{��!��陚?�z�AA��Cz�                                    Bx�[y  �          @����B�\��Q�?�{A!�Cv���B�\�޸R?�{A?
=Cvc�                                    Bx�[��  �          @����33��33?˅A@��C|
=�33��G�?�A_�
C{�f                                    Bx�[�^  �          @��Q���\)?��RA733C{!H�Q���p�?�  AV�\Cz�q                                    Bx�[�  e          @��H���ᙚ?�33A*{C{s3����  ?�33AI�C{O\                                    Bx�[��  �          @�33�����?fff@ٙ�C{޸����
?�z�A��C{Ǯ                                    Bx�[�P  T          @�=q�E���p�?��RA1��Cu�R�E��ۅ?޸RAP��CuǮ                                    Bx�[��  e          AG��J=q��z�@0��A�(�Cu^��J=q��G�@AG�A��Cu�                                    Bx�[ߜ  �          A����=q��\)>��?���Co����=q��
=>��@;�Co��                                    Bx�[�B  �          A{�������?:�H@�p�CkB�����Ӆ?z�H@�
=Ck�                                    Bx�[��  �          Aff��Q����
?ٙ�AAG�Ci:���Q��ə�?���A]p�Ch�                                    Bx�\�  "          A�\���
��
=@\��A���Cd5����
���H@j�HAθRCc��                                    Bx�\4  
�          A�������H@N�RA��\Ce��������R@]p�A�{Cd޸                                    Bx�\(�  T          @�ff�:�H���?��RA.�\CwǮ�:�H���H?�\AO�
Cw��                                    Bx�\7�  "          @�ff�'����
>�z�@ffCzk��'���33?z�@�\)Cz^�                                    Bx�\F&  T          @�p���33���
=���HC��H��33��(���z��
=C��                                    Bx�\T�  �          @���k�����>aG�?˅C��׿k���G�?�@w�C��H                                    Bx�\cr  �          @�=q�Y�����׾\)��  C���Y������>8Q�?�ffC��                                    Bx�\r  T          @�p��
=��z�>8Q�?���C��Ϳ
=���
>��H@i��C���                                    Bx�\��  T          @��R�������z���\)C�9�����=q������C�:�                                    Bx�\�d  T          @��ͿE����H�#�
����C�5ÿE����H>.{?�(�C�5�                                    Bx�\�
  �          @�ff�fff��G��k���{C��R�fff��=q�����=qC��)                                    Bx�\��  �          A (����R���}p����
C�⏾��R��
=�(����\)C���                                    Bx�\�V  T          @�{>��񙚿��\���C��q>���33�u����C���                                    Bx�\��  �          @�\?�����H�p����RC��?�������s\)C�t{                                    Bx�\آ  
�          @���?У����
�1���\)C�*=?У��Ϯ� ����ffC��                                    Bx�\�H  �          @�  ?�
=��{�%��  C���?�
=�ə��z���G�C��                                    Bx�\��  �          @�?�p����H�,������C�� ?�p���ff��H����C��                                     Bx�]�  �          @��
?��R�ƸR�HQ�����C���?��R���H�7
=���C��q                                    Bx�]:  �          @��H?�\)���
�hQ����RC���?�\)�����XQ���p�C���                                    Bx�]!�  �          @߮?������\)�"  C���?�����
��Q���\C���                                    Bx�]0�  �          @�{�5���ͽ��Ϳ:�HC�B��5����>k�?��
C�AH                                    Bx�]?,  �          @�Q�#�
��>�  ?�p�C�]q�#�
��
=?�@�Q�C�]q                                    Bx�]M�  �          @�=q>�
=���ÿ   �x��C���>�
=��G��.{���C��H                                    Bx�]\x  �          @�\?���\)�h������C�=q?���Q�
=��33C�:�                                    Bx�]k  �          @��>�G����ÿ�{��RC��\>�G���=q�J=q��33C���                                    Bx�]y�  �          @�{�B�\���
�Tz��ӅC�>��B�\���Ϳ   ����C�>�                                    Bx�]�j  �          @�{?�Q����
�Mp��	�RC���?�Q���Q��@���   C�C�                                    Bx�]�  �          @�G�?\�����*�H���C�\?\��Q��p���33C�ٚ                                    Bx�]��  �          @��?���ff��{�z�C���?��׮�L����ffC��                                    Bx�]�\  �          @�G�>�G��陚?���A#�
C���>�G���\)?�z�AM�C���                                    Bx�]�  �          A녿G���@j=qA��HC����G���  @\)A�  C��{                                    Bx�]Ѩ  �          A\)�����@e�A���C�޸�����{@z�HA�=qC���                                    Bx�]�N  �          A녾\)��(�@��A��C�� �\)��Q�@1G�A�
=C�}q                                    Bx�]��  �          @�Q�O\)���
@�B��C�ÿO\)��z�@�ffB)�\C��                                    Bx�]��  �          A�\���\��(�@�  Ba=qC�y����\���@�ffBkC�3                                    Bx�^@  �          A
=�0���333A\)B�=qC�!H�0���=qAG�B�z�C~#�                                    Bx�^�  
�          AQ쾣�
�H��@�B��fC�����
�1�@�  B�L�C��\                                    Bx�^)�  �          A�þ8Q��aG�@�Q�B��C���8Q��I��@�p�B��\C�b�                                    Bx�^82  �          A
�\�����   A=qB�.Cm�
�����ffA  B��3CiG�                                    Bx�^F�  �          AQ쿧��8��AffB�z�Cu�)����\)Az�B�k�CrL�                                    Bx�^U~  �          A��=q�W�A�B���C|B���=q�>{AQ�B�.Cz{                                    Bx�^d$  �          A���=q�0��AffB�Cx�Ὴ=q��Az�B�#�Cu��                                    Bx�^r�  �          A
=�Q��vff@��B|=qC��)�Q��^{@�p�B��\C�XR                                    Bx�^�p  �          A
{�
=q��z�@�\B\�
C��{�
=q��G�@��Bg�C��
                                    Bx�^�  �          A
=�����
@��BN�\C��=������@��BY�RC�`                                     Bx�^��  �          A�=p���
=@�z�BSp�C��=p���z�@��
B^�C�w
                                    Bx�^�b  �          A�R�G����@�=qBK��C��H�G���  @��BV��C�XR                                    Bx�^�  �          @�(��#�
���@�p�B�C�J=�#�
���H@�\)B  C�C�                                    Bx�^ʮ  �          A���  ��ff@���B�HC�׾�  ��ff@�  B33C���                                    Bx�^�T  
�          Az����ڏ\@�
=B=qC��{����љ�@���B%��C���                                    Bx�^��  �          A\)��\��ff@�
=Bp�C�⏿�\��@��B�
C��                                    Bx�^��  �          A�׿+���@�Q�BffC�%�+�����@��HB#C�f                                    Bx�_F  �          Aff�   ��Q�@���BC��Ϳ   �Ǯ@�33B'33C��{                                    Bx�_�  �          A{������H@�{B"{C��q������@�  B-��C��                                    Bx�_"�  �          A��z��ʏ\@��
B%�C�ff�z���G�@�B1p�C�Ff                                    Bx�_18  �          A	p�����33@�p�B&�\C�n�����@�\)B2�C�N                                    Bx�_?�  �          A�
�����G�@���B/�C��R�����\)@˅B:�RC���                                    Bx�_N�  �          A	�����
=@�p�B7��C���������@�
=BC33C��                                    Bx�_]*  �          AG�����{@ᙚBg
=C�4{�����\@��Br�C�"�                                    Bx�_k�  �          A{��Q���(�@��Be  C�ff��Q�����@��Bp�C�Y�                                    Bx�_zv  �          A�>W
=����@�Q�Bc��C�c�>W
=����@�\)BoG�C���                                    Bx�_�  �          A Q�>�p����@�p�Bm��C���>�p��l��@��
By��C���                                    Bx�_��  T          @��R?O\)�/\)@�z�B���C�=q?O\)�
=@��B�B�C�}q                                    Bx�_�h  �          @�ff>k����H@�p�B_��C���>k���  @�(�BkffC���                                    Bx�_�  �          A
=�
=q��@�33B,�
C�j=�
=q��z�@���B8��C�G�                                    Bx�_ô  �          A�
�n{�Å@��B,��C��\�n{����@��
B8�C�u�                                    Bx�_�Z  �          A	녿\(���=q@���B2z�C��q�\(���  @˅B>=qC��                                    Bx�_�   �          AQ�p����G�@�33B8\)C�h��p����
=@�z�BD�C�#�                                    Bx�_�  �          A33<#�
��ff@���BL�RC��<#�
��33@ᙚBX��C��                                    Bx�_�L  T          AG��\)��\)@��BN=qC�H��\)���
@�{BZ33C�<)                                    Bx�`�  �          A��������@�BW{C��q������@�  Bc  C���                                    Bx�`�  �          AQ����Q�@�ffB\Q�C�` �����
@�ffBh=qC�(�                                    Bx�`*>  �          A(���\���R@�BhC��R��\��=q@�RBt�C�h�                                    Bx�`8�  �          A녿
=����@�=qBw\)C��
=�n�RA z�B���C�~�                                    Bx�`G�  �          Aff����@�33Bw{C�b����p��A ��B�� C��)                                    Bx�`V0  �          A  �=p��~{@�Q�Bz
=C��{�=p��c�
@��RB��C��                                    Bx�`d�  �          A�׿p���W�@�G�B��C~xR�p���<��@��RB�u�C|h�                                    Bx�`s|  �          AQ�aG��QG�@���B��C~���aG��7�@�{B�Q�C|�                                    Bx�`�"  �          A��z��2�\@�=qB��C���z��Q�@�ffB���C��                                    Bx�`��  �          Aff�J=q�7�@��B��\C~�\�J=q���A   B�W
C|�                                    Bx�`�n  �          A(��333�*=q@���B�
=CY��333�  @��B��)C|��                                    Bx�`�  �          A33�W
=�.{@�{B�p�C|�f�W
=��
@�=qB�.Cz{                                    Bx�`��  �          A���L���>{@�B�L�C~�ÿL���$z�@�z�B��C|�                                     Bx�`�`  �          A �ͿaG��?\)@�p�B�ffC}���aG��%@�=qB�.C{.                                    Bx�`�  �          A �ÿO\)�QG�@��B�� C�3�O\)�8Q�@�\)B�aHC~Y�                                    Bx�`�  �          A(��:�H�qG�@�G�Bx�
C��=�:�H�XQ�@�\)B�aHC���                                    Bx�`�R  �          A��&ff�UA	�B��RC�t{�&ff�8Q�A�B��C��R                                    Bx�a�  �          A���@  �UA�HB���C����@  �8��A	��B��qCn                                    Bx�a�  �          A�R�L���EA	p�B�B�C���L���(Q�A�
B��C}�                                    Bx�a#D  �          A���  �b�\A	�B�B�C��þ�  �Dz�A��B�W
C��f                                    Bx�a1�  �          A33�B�\�h��A�B���C����B�\�J�HAffB��qC�J=                                    Bx�a@�  �          A�>Ǯ��ffA��B|\)C���>Ǯ�p  AQ�B�=qC��                                    Bx�aO6  �          A�\�s33�33A��B�ffCw�Ϳs33��A\)B��CrT{                                    Bx�a]�  �          A����׿z�HA�B��COh���׾�AffB�G�CA�{                                    Bx�al�  �          A�H���Ϳ�{A
=B��{C\T{���ͿW
=A  B�CO��                                    Bx�a{(  �          A�R�����
A=qB��Cr�)�����ffA�
B�33Ck�f                                    Bx�a��  �          AG����\���Az�B�k�Cu{���\��Q�A{B���Cn�
                                    Bx�a�t  �          A{��  �p�A�B���Cu�쿀  ���HA�HB�B�Co�
                                    Bx�a�  T          Aff��\)�
=qAG�B���Cr�H��\)��z�A
=B�\Cl�                                    Bx�a��  
�          A
=�s33��AB�z�Cwuÿs33���
A�B�  Cq��                                    Bx�a�f  �          A  �&ff�(Q�A�B�ǮC�f�&ff��A  B���C|�                                    Bx�a�  �          A�
�B�\�,(�AG�B���C~O\�B�\�(�A\)B�z�Cz�R                                    Bx�a�  �          A\)�=p��%A��B��RC~��=p��A
=B��Cz��                                    Bx�a�X  �          A(��:�H�#33A{B���C~\�:�H��
A(�B�k�Cz��                                    Bx�a��  �          A
=�8Q��%�A��B�C~s3�8Q��ffA�RB��
C{
                                    Bx�b�  �          Az�Q��FffA�B��3C(��Q��'�A�B��\C|�
                                    Bx�bJ  �          A�\�(���q�A��B�{C��(���S�
A��B�\C�]q                                    Bx�b*�  �          A�H��R����A�B��\C����R�l(�A�HB��{C�33                                    Bx�b9�  �          A�0����\)A	B~�C�XR�0���qG�A�B�\C���                                    Bx�bH<  �          A�ÿ@  ���A33Bxp�C�%�@  �|��A
�RB�.C���                                    Bx�bV�  �          A{�h�����HA��BzQ�C��h���xQ�AQ�B�{C�c�                                    Bx�be�  �          Ap��c�
����AffBt�C�uÿc�
���\A
{B�ffC��q                                    Bx�bt.  �          A�Ϳh����{A�Bi{C��h����  A�Bt��C�C�                                    Bx�b��  �          Az�5����@�\)Bc�RC��5��
=A�
Bo�C��{                                    Bx�b�z  �          A\)����c�
A�B�� C��q����E�AffB�� C�=q                                    Bx�b�   �          A\)�\)�4z�A��B�.C���\)��A�HB�33C�@                                     Bx�b��  �          A����R�%A�B��HC������R�ffA  B��)C���                                    Bx�b�l  �          A������   A=qB��C�@ �����   A(�B��fC�U�                                    Bx�b�  �          A=q��{��\Ap�B�.C����{��ffA33B��C��H                                    Bx�bڸ  �          Aff�&ff�{A��B�Q�CO\�&ff��p�A�RB�{C{�H                                    Bx�b�^  �          A�׿Q��33Az�B�B�Cx{�Q녿ǮA{B��RCr#�                                    Bx�b�  �          A�\�\)���A�B�ǮC�\)���HA�B��C{��                                    Bx�c�  �          A(���\�   A�RB���C�5ÿ�\� ��A��B�z�C��                                    Bx�cP  T          A�
��  �
=A\)B�33Ct�
��  ��\)A��B�p�Cn33                                    Bx�c#�  �          A�׿p����Az�B��\Cu��p�׿��
A�B���Cnp�                                    Bx�c2�  �          Az���  A�B�
=C�.����\AG�B��
CO\                                    Bx�cAB  �          A�H�\)��A�B��
C�AH�\)���A�B�C�Ф                                    Bx�cO�  �          A�R�B�\��
=A
=B�8RCx}q�B�\��Q�Az�B���Cr&f                                    Bx�c^�  �          A  �8Q���A��B��=Cy
=�8Q쿳33A�B���Cr�                                     Bx�cm4  �          A\)�(���(�A�B��C|�R�(���  A��B��Cwٚ                                    Bx�c{�  �          A�H��\���A{B��qC��)��\��ffA�B�p�C~)                                    Bx�c��  �          A녾�
=�33A��B�G�C�� ��
=����A�\B�C��f                                    Bx�c�&  �          A=q��{��
Ap�B�p�C�Ф��{����A
=B�8RC��\                                    Bx�c��  �          A  �L�Ϳ�ffAz�B�W
Cv��L�Ϳ��AB��=Co�                                    Bx�c�r  �          A{�\(���  A  B�\)Ck}q�\(��G�A��B�C^+�                                    Bx�c�  �          A��W
=��G�A\)B�CfuÿW
=�
=qA  B��CU{                                    Bx�cӾ  �          A(��s33��  AffB��
Cbh��s33�
=qA
=B�p�CQ^�                                    Bx�c�d  T          A\)��ff��(�A�B���Ce=q��ff�B�\A�B��qCW�                                    Bx�c�
  �          A33��녿��HA��B�(�Cb�
��녿@  Ap�B��fCUT{                                    Bx�c��  �          A���  ��A�\B���COk���  ��\)A�RB�=qC7�3                                    Bx�dV  �          Ap���=q�   A(�B��HCL�f��=q�L��AQ�B�G�C6�                                     Bx�d�  �          A����녿.{A  B�(�CR�ῑ녾k�AQ�B�B�C?c�                                    Bx�d+�  �          A{�������A  B���Cb!H����&ffA��B�ffCS�
                                    Bx�d:H  �          A33�J=q��p�A��B�  Cq���J=q���
A��B�ǮCh:�                                    Bx�dH�  �          A=q�Tz��  A�\B�{Ct�TzῦffA�B�\Cm��                                    Bx�dW�  �          A�\��\�j=qA
=B�8RC����\�L��AB���C�s3                                    Bx�df:  �          A{�(��O\)A��B�  C��{�(��1�A\)B�� C�ٚ                                    Bx�dt�  T          A Q�333�J=qA�B�Q�C��)�333�,(�A�B�CaH                                    Bx�d��  T          A\)���A�A�B�(�C��;��#�
AB���C��                                    Bx�d�,  �          A"�R�8Q����A\)B�B�C�@ �8Q��w
=A�\B��qC��{                                    Bx�d��  �          A$(��E��\)A�\B�L�C��ͿE��a�Ap�B��RC��)                                    Bx�d�x  �          A#33���r�\A33B��
C��R���U�A�B�L�C�#�                                    Bx�d�  �          A Q�5�y��A=qB��C��)�5�]p�A�B��C�7
                                    Bx�d��  �          A�\�}p���
=A
ffBsp�C��}p�����A�B~(�C�~�                                    Bx�d�j  �          A33�Q����HAz�Bx
=C�녿Q���p�A�
B�aHC�j=                                    Bx�d�  T          A��������A(�B�{C��{����u�A33B�z�C�y�                                    Bx�d��  �          A33�   ���RA=qB|�C�˅�   ����Ap�B��
C�u�                                    Bx�e\  �          A�R�Y���n{AffB��3C��=�Y���R�\A�B��Cp�                                    Bx�e  �          A�R�0���h��A33B�{C��=�0���Mp�AB�W
C���                                    Bx�e$�  T          A��E��z=qA{B��)C�lͿE��^�RA��B��C�                                    Bx�e3N  �          A\)����o\)A\)B�8RC�s3����S�
A�B�u�C�޸                                    Bx�eA�  �          Aff�.{�{�A��B��C��.{�`  A�B��qC��f                                    Bx�eP�  �          Aff�J=q���\A�B�\)C����J=q�j=qA�\B��C��                                    Bx�e_@  �          A녿^�R��33A�HB��qC��)�^�R�k�AB��)C�U�                                    Bx�em�  �          A33���\����A��B��qC�\���\�fffA�B���C~@                                     Bx�e|�  �          A
=�s33�x��A��B��=C�&f�s33�^{AQ�B��\C~�R                                    Bx�e�2  T          A�\���}p�A�B�CzQ쿵�c33A=qB���CxB�                                    Bx�e��  �          A{��Q��~�RA=qB��Cv�q��Q��e�A��B�ǮCt��                                    Bx�e�~  T          A녿�ff����AG�B~
=Cuٚ��ff�g�A(�B���Cs�=                                    Bx�e�$  �          Ap���\)�}p�A��B~=qCt����\)�c�
A�B��CrJ=                                    Bx�e��  �          A�R�333�XQ�A��B�W
C�*=�333�=p�A
=B�G�C�`                                     Bx�e�p  T          A�ÿ!G��UA33B�z�C��Ϳ!G��;�Ap�B�k�C��{                                    Bx�e�  T          A33�����N{A=qB��)C�xR�����4z�AQ�B���C���                                    Bx�e�  �          A(��W
=�QG�A
=B��RC�/\�W
=�7�A�B��C��\                                    Bx�f b  T          A��Ǯ�QG�AffB�k�C��)�Ǯ�7�A��B�Q�C�&f                                    Bx�f  �          A(��\�U�A�\B��
C����\�;�A��B��RC�J=                                    Bx�f�  �          A  �\)�N�RA�HB��C�!H�\)�5A��B�z�C�xR                                    Bx�f,T  �          A�����N�RA(�B���C�0�����5A{B�C���                                    Bx�f:�  �          A�����H�L��A��B���C������H�333A�HB�k�C�                                    Bx�fI�  �          A��ff�J�HA�B�\C�  ��ff�1�A33B�ǮC�q�                                    Bx�fXF  �          A!p�<#�
�|(�Az�B���C��<#�
�c33A
=B��qC�3                                    Bx�ff�  �          A!G����dz�A�\B�{C�K����K�A��B�C��R                                    Bx�fu�  �          A!���=p��P��Az�B�\)C��R�=p��7
=AffB��C�                                     Bx�f�8  �          A!p��z�H�HQ�Az�B�C|�=�z�H�/\)AffB�k�Cz:�                                    Bx�f��  �          A �Ϳ���L��A�B��
C|����4z�Ap�B�8RCy                                    Bx�f��  �          A z�J=q�L(�A�B�C���J=q�333Ap�B�33C~5�                                    Bx�f�*  �          A �׿u�0  A��B�L�Cz�q�u�
=A\)B��\Cw��                                    Bx�f��  �          A Q�c�
�"�\AffB�ǮCz���c�
�
=qA�
B�  Cw��                                    Bx�f�v  �          A (��^�R�ffA�
B�CwY��^�R���HA�B��
Cr�3                                    Bx�f�  �          A   ��ff�
=A\)B���Cs}q��ff��p�Az�B��\Cn��                                    Bx�f��  �          A�����\)A{B��
Co�쿧���\)A\)B��\Cj�{                                    Bx�f�h  �          A�H�����{AB��\Cq�
��������A
=B�Q�Cm                                    Bx�g  �          AG�����A��B�u�Cp쿕��z�A�B�#�Cj��                                    Bx�g�  �          A Q��\�
=A��B�Ci!H��\���RA=qB��CdxR                                    Bx�g%Z  �          A!p������A��B���Cc�����A=qB���C_�                                    Bx�g4   �          A �Ϳ��R���A
=B��C]�H���R��
=A  B�CW��                                    Bx�gB�  �          A �Ϳ����
=A�
B���C_&f�������A��B�W
CXn                                    Bx�gQL  �          A!���p����AG�B�{CY#׿�p��s33A�B�W
CP�)                                    Bx�g_�  �          A   ���ÿ��A�B��
CX\)���ÿ}p�Az�B�
=CP��                                    Bx�gn�  �          A�Ϳ������A�RB��)Ct)���ÿ�\)A�
B�z�Cp0�                                    Bx�g}>  �          A�Ϳ(��6ffAB�u�C��q�(�� ��A33B�ffC�9�                                    Bx�g��  �          A�;��8Q�AB�ffC�z���#33A33B�\)C���                                    Bx�g��  �          A�þ�ff�G�A��B�u�C��f��ff�333A=qB�aHC�o\                                    Bx�g�0  �          A(���R�-p�A��B��C����R���A
=B��RCs3                                    Bx�g��  �          AQ��R�8Q�A��B��
C��ÿ�R�$z�AffB���C�=q                                    Bx�g�|  �          A�H��  ��RAG�B�G�C�!H��  ��A�\B��C���                                    Bx�g�"  �          A33����:�HA�
B��HC�n����'�AG�B���C�"�                                    Bx�g��  �          A33�#�
�Y��A��B��
C����#�
�G
=A�RB��{C���                                    Bx�g�n  �          A��<#�
�@��A��B�33C��<#�
�.{AffB��fC�                                    Bx�h  �          A��>W
=�S33A33B�ffC�Ǯ>W
=�AG�A��B�\C���                                    Bx�h�  �          A�>�  �J�HA
�\B���C�5�>�  �9��A(�B�G�C�j=                                    Bx�h`  �          A�>����8Q�A(�B�=qC���>����'
=Ap�B�ǮC�h�                                    Bx�h-  �          A
=>8Q��!G�AG�B�\C��>8Q��  AffB��{C�G�                                    Bx�h;�  �          A��>Ǯ�#�
A�B��C�Y�>Ǯ��\A��B�aHC�ٚ                                    Bx�hJR  �          AG�?\)�0��A
{B��
C���?\)�   A\)B�8RC�J=                                    Bx�hX�  �          A�?J=q�5�A	G�B�=qC�Ф?J=q�$z�A
�\B�� C���                                    Bx�hg�  �          A33?c�
�4z�A33B�ffC��
?c�
�$z�Az�B���C��                                     Bx�hvD  �          A33?B�\�4z�A\)B��C��{?B�\�$z�A��B��C�E                                    Bx�h��  �          A��?8Q��FffA�B��HC��?8Q��6ffA��B�\C�
                                    Bx�h��  �          A��?Y���A�A�B�8RC��{?Y���333A��B�Q�C�w
                                    Bx�h�6  �          Aff?���Dz�Az�B�
=C�Ф?���5AB�C���                                    Bx�h��  �          A�H?�
=�Mp�A  B�  C�"�?�
=�>�RAp�B��C���                                    Bx�h��  �          A\)?��
�N{A��B��3C��
?��
�@  A{B���C�t{                                    Bx�h�(  T          A��?�
=�Tz�A=qB�p�C�˅?�
=�EA�B�G�C�q�                                    Bx�h��  �          Az�?��
�Y��A��B��3C�e?��
�K�A=qB��\C��                                    Bx�h�t  �          Az�?��
�\(�A��B��C�]q?��
�N�RA�B��fC��                                    Bx�h�  �          A�?!G��G
=AffB���C��
?!G��9��A�B�p�C��                                    Bx�i�  �          A��?h���H��A�B��HC��?h���;�A��B���C���                                    Bx�if  �          A��?���AG�A�
B�(�C��{?���3�
A��B��HC���                                    Bx�i&  �          A�?h���A�A=qB��C�]q?h���5�A\)B�L�C���                                    Bx�i4�  �          A?��
�_\)A Q�B�Q�C�R?��
�S33A��B��
C���                                    Bx�iCX  �          A��?��
�h��@��B�C���?��
�]p�@��B�p�C�                                      Bx�iQ�  T          A{?�  ��G�@���Bv33C���?�  �w
=@�z�B{�C��\                                    Bx�i`�  �          A�?��H���
@�  Bt�C�(�?��H�|��@��HBx��C�~�                                    Bx�ioJ  �          A��?O\)��\)@�Br��C�ff?O\)���@���Bw�
C��q                                    Bx�i}�  
�          A(�?��R��G�@�B\(�C��H?��R��z�@��B`C��                                    Bx�i��  �          A�
?�(����@���BH��C�5�?�(�����@�(�BMQ�C�p�                                    Bx�i�<  �          Aff?�(����@�(�BF�C�t{?�(���@ϮBJ��C��)                                    Bx�i��  �          A	�?L������@�BV33C���?L����p�@���BZ�C���                                    Bx�i��  �          A��>�����\@��Bt�RC��>����p�@��By�C�.                                    Bx�i�.  �          A(�<��L��A�\B�Q�C�J=<��B�\A�B�z�C�N                                    Bx�i��  �          A33>�(��|(�@��Bz�\C�q>�(��s33@�\)B~�RC�:�                                    Bx�i�z  �          A	�>�����
=@�RBc  C�˅>������H@陚Bg{C��R                                    Bx�i�   �          A��>����z�@�ffBg��C���>����Q�@���Bk�C���                                    Bx�j�  �          A��>L����z�@��
Bp{C�L�>L����Q�@�{Bs��C�W
                                    Bx�jl  �          A(�=u���@��
Br  C�b�=u��{@�{BuC�e                                    Bx�j  �          A	>�z��|(�@�{B{G�C�  >�z��tz�@�Q�B~�C�1�                                    Bx�j-�  �          A�>.{�l��@�
=B~z�C�H�>.{�e�@��B�  C�S3                                    Bx�j<^  �          A{�8Q��<��@�\B�u�C�:�8Q��5�@�(�B�.C�(�                                    Bx�jK  �          A z���.{@�G�B�L�C������'
=@�\B���C���                                    Bx�jY�  �          Ap�=�\)�@��@�B�#�C��=�\)�:=q@���B�C��=                                    Bx�jhP  �          A��u�E�@��HB���C��׾u�>�R@�(�B�33C���                                    Bx�jv�  �          A{�J=q���@��RB�G�Cp0��J=q���
@�\)B��\Cn=q                                    Bx�j��  �          Az������A ��B�\C~ff����  A ��B�k�C}�\                                    Bx�j�B  �          Az�#�
��
=A�B��Cy��#�
��=qAp�B�.Cw޸                                    Bx�j��  �          A�
��Q��G�@�(�B�C��ͽ�Q���@���B�{C��H                                    Bx�j��  �          A��>�\)�W
=@�G�B���C�^�>�\)�Q�@�\B�
=C�n                                    Bx�j�4  �          A>����@�z�BF��C���>����H@�{BIQ�C��                                    Bx�j��  �          A�>�Q���ff@���B)�C��
>�Q���z�@�
=B+�C���                                    Bx�j݀  �          A
=�W
=��  @���BI��C�⏾W
=��{@ָRBKC��                                     Bx�j�&  T          A��B�\��\)@�Q�BT��C���B�\��p�@��BV��C��
                                    Bx�j��  �          A�ͿB�\�0��A (�B��3C~�=�B�\�,(�A z�B���C~.                                    Bx�k	r  T          A	��:�H�S33@��B�� C��Ϳ:�H�O\)@�(�B�\)C��\                                    Bx�k  �          A��>����\)@�{Bi�C��R>����p�@�\)Bk(�C��)                                    Bx�k&�  �          A�#�
�z=q@��B}�\C�` �#�
�w
=@�B{C�O\                                    Bx�k5d  �          Aff�u�o\)A   B�aHC�\�u�l��A Q�B�\CaH                                    Bx�kD
  �          A��p���x��A (�B}��C�9��p���vffA z�B
=C�%                                    Bx�kR�  �          A�\�0�����A{B{\)C�:�0����=qAffB|z�C�.                                    Bx�kaV  �          Ap��z�H��\)@��Bd��C�n�z�H��{@�(�Be��C�c�                                    Bx�ko�  �          A�H�xQ�����@��BY�C���xQ���  @�(�BZ��C�H                                    Bx�k~�  �          A{�J=q��
=A	�Bs�C�<)�J=q��{A
=qBt�C�4{                                    Bx�k�H  �          A&{�L�����ABk33C����L������A�Bk��C��{                                    Bx�k��  �          A%��Q���
=A��Bl�C��=�Q���ffABm33C��f                                    Bx�k��  �          A&ff���
��z�A��Bh�C������
���
A��Bhz�C���                                    Bx�k�:  T          A%�}p���33A\)B[�
C�7
�}p����HA�B\{C�5�                                    Bx�k��  �          A�\�W
=����@�\)BV��C�޸�W
=����@�\)BV�C�޸                                    Bx�kֆ  �          Az�E����A
�RBzC��E����A
�RBzC��                                    Bx�k�,  T          Ap������33A33BxffC�c׾����33A33Bx=qC�c�                                    Bx�k��  �          A!�333����A\)Bo  C��333��G�A33Bn�RC��                                    Bx�lx  �          Ap���  ����A z�BcffC����  ��p�A Q�Bc  C���                                    Bx�l  �          A�
��G����
@�=qBr��C���G���z�@��Br
=C�q                                    Bx�l�  T          A(�������@��Bj�RC��������H@���Bj
=C�                                      Bx�l.j  �          A�H�#�
���@�=qBM�\C�ٚ�#�
���\@�G�BL�RC��q                                    Bx�l=  �          A�\��{����@�
=BA��C�w
��{��@�ffB@��C�y�                                    Bx�lK�  �          A����
=��(�@�33B;��C�R��
=���@�=qB:�HC��                                    Bx�lZ\  �          AQ�\�ָR@׮B4Q�C�ff�\��  @ָRB3�C�h�                                    Bx�li  T          A33>���33@ϮB-C��>���z�@�ffB,ffC���                                    Bx�lw�  �          A�?�p���  @�B"�\C�/\?�p���G�@�z�B!{C�&f                                    Bx�l�N  �          A�
?333��ff@��RB=qC���?333��\)@��B��C���                                    Bx�l��  �          A�
>�����@���A�
=C��)>������@��A�p�C���                                    Bx�l��  �          A�H��z���Q�@�  B#��C�ῴz����@�ffB"{C�%                                    Bx�l�@  �          A=q��p���p�@޸RB2  C��
��p���\)@�z�B0  C�                                    Bx�l��  �          A=q��z���33@�=qB*��C~k���z����@�  B(��C~��                                    Bx�lό  �          A#33����(�@��B.G�C~�ÿ���ff@�ffB,  C�                                    Bx�l�2  �          A,�׿�33����@�33B:�RC&f��33��@���B8Q�CQ�                                    Bx�l��  �          A-G�������ff@�(�B:�C�Ϳ�����G�@�G�B8�C��                                    Bx�l�~  �          A.{��Q���{@�B;(�C~޸��Q���G�@��HB8z�C\                                    Bx�m
$  �          A)�
=��33@��B?Q�C|޸�
=��ff@�=qB<�C})                                    Bx�m�  
�          A0�������HA�BC�Czu�����ffA�B@��Cz�                                     Bx�m'p  �          A0  �ff���A��BD\)Cz��ff��p�A�BAQ�C{8R                                    Bx�m6  �          A0z��
=��Q�A�B<Q�C#׿�
=��(�@�
=B9{C\)                                    Bx�mD�  �          A/
=�  ��33A Q�B<�C|n�  ��
=@��B9(�C|�3                                    Bx�mSb  �          A.�H�(���p�@�B:ffC|�q�(��陚@�=qB6�HC}E                                    Bx�mb  �          A.=q��\��\)Ap�B@33C}���\��@�
=B<�\C}�R                                    Bx�mp�  �          A0(�����{A
=BHC{����ڏ\AG�BE  C|G�                                    Bx�mT  �          A2�R�У�����@�33B2��C�ÿУ����@��RB.��C�/\                                    Bx�m��  �          A0�Ϳ�
=��Q�@���B/��C�⏿�
=����@�Q�B+�C��q                                    Bx�m��  �          A.�H��z��{@���B�C�Ϳ�z��Q�@�\)BQ�C�)                                    Bx�m�F  �          A.=q�fff�	��@ҏ\B
=C�׿fff��@��B�C�\                                    Bx�m��  �          A.�R��z���H@�33B"��C��{��z���@�B��C�                                    Bx�mȒ  �          A(  ��\)��ff@�Q�B/(�C����\)��33@�33B*ffC��                                    Bx�m�8  �          A���{���\A�Bd��Cyٚ��{��Q�A(�B_�Cz�                                     Bx�m��  �          AG���33�P��A33B��Co�\��33�]p�A	�B��HCq=q                                    Bx�m�  �          A"�\��\���A  Bz��Cw����\����AffBu��Cx��                                    Bx�n*  �          A"ff�	������AG�Bq�
Ct���	����\)A�Bl�HCu�{                                    Bx�n�  �          A"�\�����A��BrffCr(�����(�A�
Bmp�CsE                                    Bx�n v  �          A#33�p����A��BpCt��p�����A�
Bk�\Cu+�                                    Bx�n/  �          A"�\�ff��p�A=qBt��Ct�)�ff����AQ�BoG�Cu��                                    Bx�n=�  �          A��z���\)A
=qBq(�Cu#��z���ffAQ�Bk��Cv5�                                    Bx�nLh  �          A (��)����{A	�Bk��Co.�)����p�A
=Bf=qCps3                                    Bx�n[  �          A!���
=����A{Bw=qCs�
=����A(�Bqz�Cu�                                    Bx�ni�  �          A!p��Q�����A33By��Cr���Q����AG�Bs��Ct(�                                    Bx�nxZ  
�          A ���z���=qA(�BgQ�CtO\�z���=qA�BaG�Cuh�                                    Bx�n�   �          A
=�\)���@��BV�Cu
=�\)���H@��\BO�HCu�R                                    Bx�n��  �          A!���Q�����A  Be\)Ct{�Q����A��B_  Cu8R                                    Bx�n�L  �          A��z���p�Ap�Bc��Ct���z���A
=B]�Cu�\                                    Bx�n��  �          A������HA ��B\�CuQ�����H@��
BU��Cv\)                                    Bx�n��  �          A{�����G�A��Bdp�Csn������A�B]�RCt��                                    Bx�n�>  �          A�����(�A�RBfffCu�R�����A  B_\)Cw!H                                    Bx�n��  �          A"ff�����A
{Bi�Ct����HA\)Bb
=CuT{                                    Bx�n�  �          Ap��=q����A�
BoffCp� �=q��33Ap�Bh\)CrT{                                    Bx�n�0  �          Aff�1G����A=qBfG�Cn�q�1G���33A�B_=qCpQ�                                    Bx�o
�  �          A���7
=��{A=qB_��Cn���7
=��\)@��RBX��Cp#�                                    Bx�o|  �          A��AG���z�@�
=BX�RCnY��AG���@���BQz�Co�\                                    Bx�o("  �          A Q��E��33Ap�Baz�Cl)�E���A�\BZ=qCm��                                    Bx�o6�  T          A{�=p���33A ��Bcz�CkǮ�=p����@�z�B\�Cm��                                    Bx�oEn  �          A�H�5��33A (�B_�CnW
�5���@�=qBX33Co��                                    Bx�oT  T          A��<����G�A33Bjz�Ci���<�����A z�Bc
=Ck��                                    Bx�ob�  �          A�/\)���@�\)B`p�Co8R�/\)��p�@�G�BX\)Cp�H                                    Bx�oq`  �          AG��-p���ff@�=qBQ  Crn�-p���Q�@�33BH��Cs                                    Bx�o�  �          A��333���\@�  BM
=CrJ=�333��z�@��BDz�Cs�)                                    Bx�o��  �          A�H�3�
��Q�@�G�BVffCp�R�3�
���H@�=qBM��Cr=q                                    Bx�o�R  �          Aff�$z���{A Q�Ba(�Cq=q�$z�����@��BX\)Cr��                                    Bx�o��  �          A=q�*=q��{@�p�BS  Crٚ�*=q����@�BI�CtE                                    Bx�o��  �          A�#33���@�
=BK��Cu)�#33��=q@�
=BB��Cv\)                                    Bx�o�D  �          A
=�?\)��@�ffBI
=Cq.�?\)��Q�@�{B?��Cr��                                    Bx�o��  �          A=q�6ff��G�@�33BF�Cr� �6ff���
@��HB=ffCt�                                    Bx�o�  �          A��K�����@�(�BH�Cp{�K���(�@�B?{Cq�
                                    Bx�o�6  �          A$(��g
=���A (�BM{Ck�
�g
=��{@��BC�HCm��                                    Bx�p�  	          A%G��W����
A��BVffCl��W�����A ��BM  Cn�)                                    Bx�p�  �          A%�w���@�z�B@
=Ck���w�����@�33B6��Cmc�                                    Bx�p!(  �          A&�\��ff���@���B*
=Ch33��ff���@�
=B �HCi��                                    Bx�p/�  T          A%�������\)@��B3Q�Cd@ ������33@��B*�Cf�                                    Bx�p>t  �          A&{��Q����@���B=��CcJ=��Q�����@��
B4��CeY�                                    Bx�pM  �          A'�
�������
@�Q�B?�Cb�=������Q�@�\)B6��Cd�                                    Bx�p[�  �          A#��y�����HA{BRffCg��y����Q�@��BH��Cis3                                    Bx�pjf  �          A$  �[�����A ��BO�Cm{�[���
=@�Q�BE=qCo�                                    Bx�py  T          A&ff���H��Q�@��\BC�CiaH���H��@��B8�Ck^�                                    Bx�p��  �          A$(��{����
@�z�B@\)Ck  �{�����@��B5�
Cl��                                    Bx�p�X  �          A%�����
=@�Q�BA��Ch��������@�{B7=qCj�f                                    Bx�p��  �          A(����=q���H@��HB?�ChJ=��=q����@�Q�B5z�CjY�                                    Bx�p��  T          A*ff���\��\)@��BB�\Cj�{���\��@���B7�Cl��                                    Bx�p�J  "          A+�
������{@��B?�Ck�R��������@�(�B4Cm��                                    Bx�p��  T          A)��������\@�ffB:��Ci�\�����ȣ�@��HB/�Ck��                                    Bx�pߖ  �          A"{�}p����\@�RB=�Cj���}p�����@��
B2�\Cl�f                                    Bx�p�<  �          A�
���
����@޸RB/z�Ch�)���
��@��HB$G�Cj�q                                    Bx�p��  �          A#���  ��{@�G�B>=qCh  ��  ��z�@�ffB2��Cj33                                    Bx�q�  �          A'
=���H��@��B?�\CjJ=���H����@��
B3�
Clk�                                    Bx�q.  �          A&ff��=q��
=@��
B4��Cgff��=q��p�@�  B)\)Ci�                                     Bx�q(�  �          A'\)������Q�@�\B9�Ce��������\)@�RB.Q�Ch)                                    Bx�q7z  �          A'���Q�����@�33B:=qCd�)��Q���(�@�B.��Cg�                                    Bx�qF   �          A'����R��Q�@��B8��Cb�����R��  @�ffB-�HCe.                                    Bx�qT�  �          A%G���G���  @陚B4=qCd����G���
=@�p�B(��CgE                                    Bx�qcl  T          A#�
�j=q��z�@��HB7  Co:��j=q���
@��B)�RCq�                                    Bx�qr  �          A!��h�����@��
B3�Coff�h�����
@�{B&
=Cq5�                                    Bx�q��  T          A ���aG���ff@�\B3=qCph��aG���p�@�z�B%z�Cr+�                                    Bx�q�^  
�          A Q��`  ���@�\B3�Cpk��`  ��z�@�z�B&  Cr5�                                    Bx�q�  �          A=q�g
=���H@�33B7�
CnB��g
=�ʏ\@�B*
=CpB�                                    Bx�q��  �          A!��c33��Q�@��B8�Cop��c33��Q�@�=qB*�\Cqff                                    Bx�q�P  T          A"=q��������@�=qB1Q�Cl(�������Q�@�(�B#�Cn33                                    Bx�q��  �          A$����=q��p�@�p�B0�Cl����=q��p�@�ffB"�Cn��                                    Bx�q؜  "          A%���=q����@��B'ffCk���=q��(�@�B\)Cm޸                                    Bx�q�B  T          A%���(����@�(�B-�Cl����(���=q@�z�B33Cn��                                    Bx�q��  T          A%p�������G�@��B/Q�Cmk������ٙ�@�p�B Cop�                                    Bx�r�  �          A'
=�e��z�@�p�B5G�Cp���e��{@��B&
=Cr�H                                    Bx�r4  �          A&ff�i����=q@���B5��Cp��i�����
@���B&Q�Cr                                    Bx�r!�  T          A%���(���{@�B!�Cg���(���{@�B
=Ci��                                    Bx�r0�  �          A#�������\)@�z�B)p�Ck�������׮@�(�B�\Cm��                                    Bx�r?&  T          A#
=��G���  @�ffB$\)Cgc���G���Q�@ƸRB
=Ci��                                    Bx�rM�  �          A$(���z���p�@ᙚB-ffC_�=��z����R@ӅB p�Cb�f                                    Bx�r\r  T          A#
=��\)����@�z�B2  Ceh���\)�\@�B#Ch�                                    Bx�rk  �          A!���������\@�33B;�CfxR�������@���B,z�CiT{                                    Bx�ry�  �          A"{��\)����@�33BB�Ca33��\)���
@�{B4p�Cd��                                    Bx�r�d  
�          A(Q������H@�33B"Q�Ce������(�@�=qB��Cg��                                    Bx�r�
  
l          A*�R�����z�@��B
ffCjh�������
@�{A�\)Cl&f                                    Bx�r��  
�          A/
=��33���@���B{Cg�{��33���H@���B�Ci�                                    Bx�r�V  T          A.=q��  ����@�  B5��Ciٚ��  ��G�@�ffB%��Cl}q                                    Bx�r��  �          A.�\�����@�\B0�Cn��������@�\)B�Cp�                                    Bx�rѢ  T          A0(�������p�@�z�B833Cg
�����ҏ\@�33B(p�Cj                                    Bx�r�H  �          A.�R�������HA��BO�Cb.������=qA��B@�\Cf+�                                    Bx�r��  �          A0z������(�@�ffB1�\CmaH�������@��HB Q�Co�                                    Bx�r��  T          A/33������
=@�  Bz�Cp.����� ��@��B��Cq�R                                    Bx�s:  �          A.=q������  @�  B {Cm\������33@�33B�\Co.                                    Bx�s�  �          A0z���Q����H@�33B
=Cnff��Q���@��B(�CpW
                                    Bx�s)�  �          A3\)��p���z�@ڏ\B33Cl^���p����@��
B�\CnaH                                    Bx�s8,  �          A3���p���@�{B�HCo}q��p���
@�A�
=Cq+�                                    Bx�sF�  "          A,z������@��
B�
Co������ ��@�z�A��\CqG�                                    Bx�sUx  T          A+33������
@��B9Q�Ch�
�����=q@��B'�HCk�f                                    Bx�sd  �          A)p�����{A�BQ�\C_������R@��BB  Cd�                                    Bx�sr�  �          A3��~{��\)@�=qB��Cs���~{�	��@���B�
Cu0�                                    Bx�s�j  T          A0(��p  ��G�@�\B Q�Cs��p  ��H@ʏ\BffCuY�                                    Bx�s�  �          A.=q�\)��G�@�=qB!�CqJ=�\)��{@ʏ\B��CsO\                                    Bx�s��  
�          A.�H�`  ��p�@�ffB%�Ct�q�`  �G�@�{B��Cv��                                    Bx�s�\  
�          A/��dz���z�@��B&G�Ct8R�dz���@�Q�B�RCv&f                                    Bx�s�  �          A.=q�]p���{@ڏ\B
=Cu�\�]p��G�@���B{Cwz�                                    Bx�sʨ  �          A.ff�9����H@�ffA�=qC|�f�9����@^�RA�(�C}��                                    Bx�s�N  |          A4  �z��)�@Dz�A}C�� �z��.{?��RA"�\C��                                     Bx�s��  T          A3\)��(��)�@G�A�p�C�c׿�(��.{@G�A&�\C��                                     Bx�s��  
�          A6ff���H�1G�@
=A?�C�� ���H�4(�?��H@ÅC���                                    Bx�t@  T          A8(���\)�1�@.{AZ�RC�ٚ��\)�4��?Ǯ@���C��                                    Bx�t�  
�          A5G���=q�1G�?�G�A33C��3��=q�3\)?
=@@��C��)                                    Bx�t"�  "          A4  �Ǯ�0��?�=q@ٙ�C��)�Ǯ�1�>#�
?Tz�C��                                    Bx�t12  �          A3
=��(��/
=?�ff@�ffC�����(��0Q�>\)?333C��\                                    Bx�t?�  T          A4(���Q��1�>L��?��\C��\��Q��1G��}p���G�C��=                                    Bx�tN~  T          A2�H���H�0��=�?#�
C��)���H�/�
������RC��
                                    Bx�t]$  �          A2�H��33�0�;#�
�Y��C���33�/���{��  C���                                    Bx�tk�  "          A1�����0(�=���?
=qC�@ �����/33������C�:�                                    Bx�tzp  "          A1p��G��0Q�O\)��ffC���G��-��G��(  C��3                                    Bx�t�  �          A0�ÿ
=�/���(���=qC�y��
=�,Q����J�RC�q�                                    Bx�t��  �          A9������6�R��  ��p�C�*=�����5��\��33C�%                                    Bx�t�b  �          A<�Ϳ޸R�:=q?.{@S33C����޸R�:=q�!G��E�C���                                    Bx�t�  �          A:{��Q��8��>�Q�?�G�C�쿘Q��8z�s33���RC��                                    Bx�tî  T          A:�H��33�8z�?Q�@�Q�C��=��33�8�Ϳ   ��RC��                                    Bx�t�T  
�          A4�Ϳ�Q��.�\��z���=qC����Q��*�H�)���^{C��3                                    Bx�t��  4          A5녿z�H�+�
�dz����\C�c׿z�H�$  ��G���p�C�C�                                    Bx�t�  "          A-p��0���%G��L(�����C�R�0���=q���
��Q�C��                                    Bx�t�F  �          A,�׿G��%��?\)���C��
�G��ff������C��                                     Bx�u�  h          A333��33�/���\)�p�C����33�+\)�9���qp�C��\                                    Bx�u�  �          A:ff����8Q�>W
=?��
C������7���z���Q�C��=                                    Bx�u*8  �          A<(���p��9?}p�@��C�\)��p��:ff���Ϳ�(�C�^�                                    Bx�u8�  �          A<z´p��:ff?@  @i��C�^���p��:ff�(���N{C�`                                     Bx�uG�  �          A4�ÿ�������33��
=C��3������Ϯ��C��H                                    Bx�uV*  �          A:�R������
��=q�	(�C�8R�����	����$�
C���                                    Bx�ud�            A<�ÿ�{�"{��\)��p�C�z��{�p������C�)                                    Bx�usv  �          A=���u��H��p���C�=q�u�����33�p�C���                                    Bx�u�  �          A<(����\)������
C��3�������#G�C�:�                                    Bx�u��  �          A;�
���������
C�������  �z��3p�C�XR                                    Bx�u�h  �          A;
=����z���\�z�C������ z���\�8=qC�"�                                    Bx�u�  |          A=p��k�����*ff�3C����k��mp��3\)G�C�@                                     Bx�u��  �          A?33�#�
����.�H� C��׽#�
�W��7\)33C��=                                    Bx�u�Z  �          A=p�<��
��=q�&�H�w�\C�R<��
��{�0���\C�!H                                    Bx�u�   �          A=G��aG���=q�   �f�C�H�aG���\)�+�33C��)                                    Bx�u�  �          A<Q����{�p��J��C�>�����{��
�hz�C��                                    Bx�u�L  �          A7\)��{�p���33�=qC��׿�{���\)�1�C�R                                    Bx�v�  �          A7�������
��
=��C�o\��������p��3��C��                                     Bx�v�  �          A6�\�ٙ�������
��C��f�ٙ���Q����8  C��3                                    Bx�v#>  �          A4zῪ=q��
=����EG�C��=��=q������H�c\)C�Ǯ                                    Bx�v1�  �          A2ff������
=��\�{�C��q�����e�'�� C|�
                                    Bx�v@�  �          A1p��W
=���\�33�x(�C�K��W
=�n{�$��W
C��                                     Bx�vO0  �          A5����U��,  �3Cz�)����=q�1�)Cmk�                                    Bx�v]�            A<�ͿaG��J=q�3�ffC~c׿aG��˅�8��¢\Cq                                    Bx�vl|  �          A>ff�xQ��K��6�H��C}�xQ�����<  ¢(�Cnk�                                    Bx�v{"  T          A>�\�G���G��0����C��R�G��.{�8��z�C}��                                    Bx�v��  �          A9�����ff�'�#�C��{����\���1�  C�w
                                    Bx�v�n  �          A:�H�:�H���$���w��C�/\�:�H�|(��/\)�HC���                                    Bx�v�  �          A<(��5��p��*�\��C���5�W��4  �fC��                                    Bx�v��  �          A;\)�������*�HǮC�c׿���K��3�
�HC���                                    Bx�v�`  �          A7\)�Ǯ��G��"=q�y��C�� �Ǯ�r�\�,��#�C�
=                                    Bx�v�  �          A7�>aG���(����Z�
C��{>aG����R�#��{��C�7
                                    Bx�v�  �          A8  >�=q���H�G��a�HC�:�>�=q�����&=qz�C��
                                    Bx�v�R  �          A6�R>�  �\��R�gQ�C�1�>�  ����'33L�C���                                    Bx�v��  �          A5�=L����  ��
�bz�C�:�=L�������$��  C�L�                                    Bx�w�  �          A4�׿#�
��z��p��i=qC��=�#�
��p��%��W
C��                                    Bx�wD  �          A5���Q���33��R�X�C��)��Q���� ���z��C��H                                    Bx�w*�  �          A0  ���R������F  C�H���R��
=����gp�C��                                    Bx�w9�  �          A.�R�#33��G��Q��X�HCvB��#33��p�����w�Cp�                                    Bx�wH6  �          A.ff�H����{��\�[Q�Co���H����G��{�x  Ch�                                    Bx�wV�  �          A/��>�R��(��33�A�Cu�\�>�R��=q����`��Cp                                    Bx�we�  �          A.=q�������ᙚ�"  C}E���׮��\�C�Cz��                                    Bx�wt(  �          A-G��fff�����33�	�HC�{�fff��=q��(��,��C��3                                    Bx�w��  �          A,Q�ٙ����
��=q�,��C��3�ٙ�����ff�N��C�                                    Bx�w�t  �          A.=q����\)������HC�.�����z���G��/p�C�e                                    Bx�w�  �          A.�H�
=�Q������
C~���
=��\)���H�)  C}
=                                    Bx�w��  �          A/33��p��Å���C�\)����������*Q�C                                      Bx�w�f  �          A+
=��������RC�������(��
=C~)                                    Bx�w�  �          A,z��ff����=q��
=C���ff��R�����C��                                    Bx�wڲ  �          A,(���{�
�\�ȣ���HC�Z῎{�������2��C���                                    Bx�w�X  �          A-���{�Q���ff���C��
��{����Q��/��C���                                    Bx�w��  T          A,(��Ǯ����������C�` �Ǯ�33�Å�
�C���                                    Bx�x�  �          A,zῢ�\�\)��\)�G�C��ῢ�\�����H�%�RC�w
                                    Bx�xJ  �          A)�\)�\)�$z��lz�C�XR�\)��H������(�C�                                    Bx�x#�  �          A33�Mp��z�@p�Ak�
Cy���Mp����?�G�@��CzxR                                    Bx�x2�  T          A"�H�U�(�?��RA�\Cz�f�U�{��\)��
=Cz�)                                    Bx�xA<  �          A�e���@#�
Apz�Cw�{�e�?�=q@�=qCx}q                                    Bx�xO�  �          A��AG��Q�?�G�@�C{��AG���ÿ��C33C|�                                    Bx�x^�  �          Az��a��G�?��@`��CxǮ�a���Ϳh�����\Cx�R                                    Bx�xm.  �          A33�Mp��{����Q�C{)�Mp���H���H�6=qCz��                                    Bx�x{�  �          A$z��4z���Ϳ�(���33C}�R�4z��\)�5��Q�C}n                                    Bx�x�z  �          A(���,(��!���R�أ�C��,(��(��:=q��ffC~�
                                    Bx�x�   �          A#33�'
=�33��(�� ��C~���'
=��
� ���=�C~@                                     Bx�x��  �          A(�ÿ�\)�������qC~aH��\)�Tz���RG�Cw�\                                    Bx�x�l  �          A(�Ϳ�\)��33�{�R�
C|��\)���H����w��Cw�{                                    Bx�x�  �          A((�����p����H�C|p������R����mffCw�3                                    Bx�xӸ  �          A%���+���=q�����:�HCwǮ�+�������^��Cr�H                                    Bx�x�^  �          A'��������ff�V�C|�����(�����{��Cw�\                                    Bx�x�  �          A(�׿�����H�G��c�
C�~�����\)��\{C|�                                    Bx�x��  �          A-�����������H�|(�C�E�����J=q�&{��C�P�                                    Bx�yP  �          A-��>Ǯ��
=����C��)>Ǯ��H�(���)C���                                    Bx�y�  �          A,�׽��
��������C�uý��
�,���'
=�C��                                    Bx�y+�  �          A(  ���
����=qG�C�t{���
�=q�#�z�C�\                                    Bx�y:B  �          A(  �G��أ���Q��>�RC}h��G������p��e33Cy&f                                    Bx�yH�  T          A(z�����G���\)�=�RC|�R���������d=qCx�H                                    Bx�yW�  �          A$������ff��p��1G�C{8R����=q����W��Cv�q                                    Bx�yf4  �          A�������
=�љ��'33C|\�����{��{�M�Cx\)                                    Bx�yt�  �          A{�\)������\��Cy.�\)�������E
=Cu=q                                    Bx�y��  �          A	��{���R��\)�:ffCw�f�{��  ���
�`�Crp�                                    Bx�y�&  �          @ȣ׿�=q�n�R���\�H{Cw��=q�333��p��m�Cp�\                                    Bx�y��  �          @��H��=q��p��k��=qCyE��=q�[���(��D�HCuO\                                    Bx�y�r  �          @�G��^�R�g����S��C�<)�^�R�*=q��Q��|=qC{޸                                    Bx�y�  �          @˅�����
=�-p���  C~G������(��j=q�p�C|}q                                    Bx�y̾  �          @��
������=q=��
?333C�Ϳ�����\)��=q���C��                                    Bx�y�d  �          @�{��
=��  �mp��\)CG���
=�o\)��\)�C=qC|p�                                    Bx�y�
  �          @�Q�h���|(���=q�M�C����h���<(����R�v��C|�\                                    Bx�y��  �          @�33������p�����C�)�����\��Q�¬W
Ch33                                    Bx�zV  
�          @���>��R?���
=ª�fBn�\>��R?��H��\)G�B�                                    Bx�z�  �          A��>��R��(��{  C���>��R=��
�(�¯�\AZ�\                                    Bx�z$�  �          A=q?z�H���� z�£�C�=q?z�H?k��    �fB,��                                    Bx�z3H  �          A ��?!G�>#�
�   ª�RAeG�?!G�?����\Q�B��R                                    Bx�zA�  �          @�G�>.{?z�H��
=¥aHB��q>.{@�H��z���B�.                                    Bx�zP�  �          Ap�>L��?G�� ��¨� B�
=>L��@33���ǮB��                                    Bx�z_:  �          A�\�L��?��H�Q�ǮBÊ=�L��@@�����H�
B��\                                    Bx�zm�  �          A�׾k�?�Q���HB�\)�k�@QG���(��)B�                                      Bx�z|�  �          A�<�@�\� ���{B�W
<�@c33��Q�\)B�
=                                    Bx�z�,  �          A�׾�@�����(�B�� ��@l����p��}��B��H                                    Bx�z��  �          Aff��
=@\)���R��B�Q��
=@|(���\�s  B�L�                                    Bx�z�x  �          @�{?@  ���J�H�\G�C��\?@  ���aG��qC���                                    Bx�z�  �          @�
=@!G���
=�Y����p�C�U�@!G������\���\C���                                    Bx�z��  �          @�p�@��������RC�f@�ڏ\�(����C�s3                                    Bx�z�j  �          @�
=@�H��G�>8Q�?�\)C�0�@�H����G��ffC�P�                                    Bx�z�  �          A	�@#�
���?�z�@���C�@#�
��R���FffC���                                    Bx�z�  �          A  @0  ��?��A�C��3@0  � �׾���޸RC�t{                                    Bx�{ \  �          Aff@
=��=q?�=qAI��C�g�@
=� z�>k�?˅C�33                                    Bx�{  �          @�33@(�����?�\)AQ�C��)@(����33��33�$z�C��                                    Bx�{�  �          @�
=?�z����
�S33�ߙ�C�Z�?�z����\���R�(�C��                                    Bx�{,N  �          @��R>Ǯ�|����(��p(�C��>Ǯ��R��G��C�p�                                    Bx�{:�  �          A33?E����\�����C\)C��?E��~{��33�qQ�C�xR                                    Bx�{I�  �          A ��?�����(�����1p�C��H?�����(���(��_{C�f                                    Bx�{X@  �          A
=>Ǯ��33��z��9p�C��q>Ǯ��G�����h33C��R                                    Bx�{f�  �          @��
?J=q���������(�C���?J=q���\���H�W
=C��=                                    Bx�{u�  �          @�p�?8Q������  �B33C�>�?8Q��e����
�p�RC���                                    Bx�{�2  �          @�ff?���ff��G��F��C��f?��QG����H�u�C���                                    Bx�{��  �          @�33?.{���H��{�5\)C�4{?.{�aG���G��d=qC�s3                                    Bx�{�~  �          @�ff?\(���
=��z��0(�C�/\?\(��j=q�����^�HC���                                    Bx�{�$  �          @�=q?���ff��Q���C���?����R����L=qC��                                     Bx�{��  �          @��H��G�������  �p�C�o\��G����\����?p�C�P�                                    Bx�{�p  �          @��H��=q��=q�C�
�r��Cf���=q�(���QG��\CSW
                                    Bx�{�  �          @��R��G��7
=��\)�M��Cnc׿�G�����\)�tCcaH                                    Bx�{�  �          @�ff��  ���R�Y���
=C}#׿�  ��z����
�3{Cz                                    Bx�{�b  T          @�Q쿬����\)�,�����HC�
���������r�\�z�C~G�                                    Bx�|  �          @��H�������z���
=C�Zῧ���  �L�����
CE                                    Bx�|�  �          @�Q�>���������
=C��H>���=q���Z�RC��                                    Bx�|%T  �          @�  ��p�����˅����C��;�p������#�
���C���                                    Bx�|3�  �          @�G���p���=q��33���C��þ�p���p�����
C�}q                                    Bx�|B�  �          @ȣ׿#�
���ÿ�
=�TQ�C����#�
���\�,(���ffC��H                                    Bx�|QF  "          @�{�+������p����
C���+��tz��ff���C�
=                                    Bx�|_�  "          @�z�
=q�x�ÿ���l(�C��
=q�e������؏\C���                                    Bx�|n�  "          @�
=�u��
=���\�E�C�@ �u�����R��G�C��                                    Bx�|}8  T          @���p�����Ϳ����=qC���p����  �\)��{C���                                    Bx�|��  �          A�aG����>8Q�?�ffC�=q�aG���{��=q�4  C�9�                                    Bx�|��  �          A�
��  �
=?&ff@���C�q��  �{��33� z�C�)                                    Bx�|�*  �          A�;.{�(�?=p�@�
=C�n�.{��������C�l�                                    Bx�|��  �          A�þB�\�  ?��
@ƸRC�o\�B�\��
�����θRC�o\                                    Bx�|�v  �          A33��(��ff?Tz�@�=qC��{��(������  ��C��3                                    Bx�|�  �          A(���ff�\)?Q�@�=qC��
��ff��\��p����C���                                    Bx�|��  �          AG������>��H@A�C�K��������Ǯ��C�E                                    Bx�|�h  �          @k���(��b�\>\@���C��\��(��a녾�
=��p�C��                                    Bx�}  �          @W���H����?�
=A�z�CUp���H��
=?��
A�CZ�
                                    Bx�}�  �          @>{��녿p��@�
B1\)CNLͿ�녿�\)?�ffB�CW�                                    Bx�}Z  �          @0  ���
�5?��B2�HCI�f���
��\)?�Q�B(�CT:�                                    Bx�}-   �          @G����H�Ǯ?�  BGQ�CFͿ��H�8Q�?���B1��CR�
                                    Bx�};�  �          @N{�p�׿c�
@:=qB�{C_z�p�׿\@(Q�B_ffCnT{                                    Bx�}JL  �          @aG���z�@�?�B{B�p���z�?�Q�@
=B3�Cff                                    Bx�}X�  �          @N�R���
?�ff@{B<��C� ���
?�@��BQ��C#�=                                    Bx�}g�  �          ?�=q�xQ�=L��?�{BY{C1&f�xQ쾅�?�=qBSp�CB��                                    Bx�}v>  �          ?�p���(�?5>aG�@أ�C�)��(�?#�
>\A9G�C�
                                    Bx�}��  �          @ �׿Tz�?��?���BFQ�Ck��Tz�>��R?��B_=qC��                                    Bx�}��  �          ?��\��?0��?.{B�B����?�?Q�BD�C�\                                    Bx�}�0  �          @�G����
@�=q�������HB�\)���
@���?G�A(�BЙ�                                    Bx�}��  �          @�  �!G�@ə�����;
=B�z�!G�@�>.{?��
B�B�                                    Bx�}�|  �          @�=q��ff@�  �G����
B����ff@�  ?:�H@�p�B���                                    Bx�}�"  �          @�G��   @�  �����B�=q�   @�?���A(�B�Q�                                    Bx�}��  �          @��>�z�@�녿���{B�u�>�z�@�33?.{@�  B�z�                                    Bx�}�n  �          @��H>L��@�p���ff�<z�B��H>L��@�\>aG�?��B��                                    Bx�}�  �          @�R>aG�@�(��n{����B��>aG�@�z�?Q�@ə�B��=                                    Bx�~�  �          @��H>\)@�Q�h������B��)>\)@��?O\)@ʏ\B��)                                    Bx�~`  �          @�Q�=u@����Ϳ@  B�#�=u@��?�z�ALQ�B��                                    Bx�~&  �          @�ff�
=q@��������RB�
=�
=q@��?�(�A0��B�33                                    Bx�~4�  �          @�{���H@��>���@@��B�\)���H@�33@(�A���B���                                    Bx�~CR  �          @�\)�#�
@�{=�?h��B��\�#�
@�{?�
=Ag�
B��H                                    Bx�~Q�  �          @�(����@�G��#�
�L��B�uÿ��@�\?�\AW�B��                                    Bx�~`�  �          @��H��{@�  >�{@%B��Ϳ�{@�R@
=A�
=B�z�                                    Bx�~oD  �          @�=q�}p�@�\)?#�
@�=qB��}p�@�@��A�p�B�Ǯ                                    Bx�~}�  �          @陚����@�p�?E�@��HB��
����@���@p�A�p�B��)                                    Bx�~��  �          @����
=@߮?�p�A8z�B��Ϳ�
=@�{@G
=A�Q�B�L�                                    Bx�~�6  �          @�ff�Ǯ@ۅ?�p�A=��Bͨ��Ǯ@ə�@Dz�A�z�B���                                    Bx�~��  �          @����(�@��
?�{A/
=B�8R��(�@��H@=p�A�B�#�                                    Bx�~��  �          @ᙚ��G�@�p�?�33Az{B�aH��G�@���@\(�A�{Bˣ�                                    Bx�~�(  �          @�\���@���@S33A���B�G����@�@�(�B%
=Bݨ�                                    Bx�~��  �          @�33�(�@�  @���Bz�B�p��(�@p��@��HBP\)B��\                                    Bx�~�t  �          @׮�
=q@�G�@�33B��B��R�
=q@�z�@�Q�BN33B��                                    Bx�~�  �          @�p��aG�@�33@��HB2
=BȔ{�aG�@a�@��
Bh�B�                                      Bx��  T          @�p��L��@��@��RBP�\B�uýL��@2�\@�=qB�aHB�8R                                    Bx�f  �          @���?   @e�@�z�Be=qB�  ?   @ff@��HB�W
B�\                                    Bx�  �          @��H?\)@�@#�
A��B��?\)@��@k�B%  B��R                                    Bx�-�  �          @�p����R@��\@a�B#{B��쾞�R@E@�p�B[p�B���                                    Bx�<X  �          @�G��xQ�@}p�@���B<�\Bϊ=�xQ�@1G�@�\)Bs  Bڳ3                                    Bx�J�  �          @����@w�@�  B7B�� ���@1G�@��\Bp�B�G�                                    Bx�Y�  �          @�녾�{@H��@�=qB]p�B����{?�@�ffB��B�\)                                    Bx�hJ  �          @��þ�p�@6ff@�
=Bj��B���p�?���@���B�u�Bͽq                                    Bx�v�  �          @�33��
=@"�\@��RBz
=B���
=?��R@�B��)B�Ǯ                                    Bx���  �          @�ff�8Q�@!G�@�=qB~(�B�Q�8Q�?�
=@�G�B��{Bř�                                    Bx��<  �          @�
=�(��@\��@��BO��Bɏ\�(��@�R@��RB��)B��
                                    Bx���  �          @������H@^{@�G�BJ��B�{���H@33@���B��
B��                                    Bx���  �          @�G����
@]p�@r�\B9ffB�(����
@=q@���Bo��B�Q�                                    Bx��.  �          @��׿�@~�R@���B5�\B�aH��@1�@�z�Bj��B�.                                    Bx���  �          @���p�@g
=@��\BR
=B���p�@{@�=qB��qB�p�                                    Bx��z  �          @�  ��(�@@  @�=qBe��B�ff��(�?У�@��B�L�B��                                    Bx��   �          @��
��@'
=@��
Bt(�B�33��?�
=@�33B���CaH                                    Bx���  �          @�����Q�?��@�=qB�ffB�33��Q�?   @��
B�8RC\)                                    Bx��	l  �          @�33��\)@(�@�z�B~\)B�k���\)?p��@�=qB�ffC\)                                    Bx��  �          @�\)�У�?�z�@�z�B��Cn�У׾�Q�@��B��C@�\                                    Bx��&�  �          @�  ����?�ff@�ffB�W
C^����þ�(�@���B�� CCL�                                    Bx��5^  �          @�����z�?�33@���B�  C�ÿ�zᾳ33@��
B�B�CA��                                    Bx��D  �          @�녿��R?�G�@���B���C�
���R��(�@�33B��CD�                                    Bx��R�  �          @��\��?���@�z�B|\)C���=u@�=qB�W
C2T{                                    Bx��aP  �          @��H�z�@�@�33Bd=qC0��z�?�ff@�G�B���C
                                    Bx��o�  �          @���   @QG�@��BA
=B��R�   ?�p�@�G�Bk�RC�                                    Bx��~�  �          @��
�C33@�
@�
=B\  C�H�C33?W
=@�(�BvC$�3                                    Bx���B  �          @޸R�{@�Q�?�=qA���B��f�{@��@K�A�  B��                                    Bx����  �          @�Q����@�\@�33B�k�B�\����?xQ�@���B��B��                                    Bx����  �          Az���ff@ᙚ@j�HA��HB�\��ff@��
@��Bp�B�.                                    Bx���4  �          A녿���@���@�=qB$G�BՅ����@xQ�@���B\��B�q                                    Bx����  �          A����@���@�G�B+�Bم��@���@��Bc{B��                                    Bx��ր  �          A
=��{@�  ?�A*�HC
=��{@�=q@U�A��RCQ�                                    Bx���&  �          Aff��
=@�  �(��fffC  ��
=@�z�B�\��p�C                                      Bx����  �          @�ff��@�{�z�H���B���@ȣ�����  B�W
                                    Bx��r  �          @��
���@Ǯ�U��33B�Ǯ���@�z��{�J�RBҽq                                    Bx��  �          A�1�A
=q=�?:�HBר��1�A��@
=AqG�B���                                    Bx���  T          A���Mp�A33����k�B�u��Mp�A
�H@(�AT��Bܔ{                                    Bx��.d  �          A\)���\A{?:�H@��HB��H���\@�z�@8��A���B��)                                    Bx��=
  �          A�R�)��A�;��
����B�
=�)��A��@(�AK�
B���                                    Bx��K�  �          A"{��A�R�h������B���A��?�(�Az�B�=q                                    Bx��ZV  �          A#33��\A   �h������B�\��\A=q?޸RAG�B�G�                                    Bx��h�  �          A#��
=qA\)�G�����B̀ �
=qA��?�{A'�
B��)                                    Bx��w�  �          A#��Q�A�\�W
=��ffB����Q�Az�?�ffA"=qB�Q�                                    Bx���H  �          A$Q��3�
A�.{�q�B����3�
A
=?�Q�A.�\B�W
                                    Bx����  �          A�
���
A��L�����
BƳ3���
A33?�Q�A#�
B��                                    Bx����  �          A��p��A�ͿQ���(�B��p��A�?�33A��B�\                                    Bx���:  �          A
=���AQ�#�
�j�HB����Ap�?�(�A7�B�\)                                    Bx����  �          A!��
=A����.�RB�\�
=Ap�@��AC�BϮ                                    Bx��φ  �          A#
=��A�R�����\B����A�R@\)AK�BΏ\                                    Bx���,  �          A!��AG��   �5�B����A��@Q�AC
=Bή                                    Bx����  �          A����H@�  >�z�@�B׳3��H@��@G�A�{B�Q�                                    Bx���x  �          @�p���=q@z=q?��
A��C	�f��=q@R�\@0��A�\)C��                                    Bx��
  �          @�
=�E�@�=q����33B��)�E�@���?�{AF{B�\)                                    Bx���            Az��_\)@��;��R�G�B���_\)@�\)?�ffA8��B���                                    Bx��'j  
�          A33�{Aff�����5�B�{�{A�?��@j=qB҅                                    Bx��6  T          A�\�#33A	녿�=q��RB��#33A
�\?��
@�Q�B��H                                    Bx��D�  �          A�����A
ff���R�   B��f���A�R@G�AP(�BӸR                                    Bx��S\  T          A����p�A�R?k�@�33Býq��p�Ap�@R�\A�ffB���                                    Bx��b  
�          A�׿�
=@��R�:=q��Q�B�aH��
=@��ÿ��\�3\)B�G�                                    Bx��p�  �          A  �33Aff������B�8R�33A
=�ٙ��(  B���                                    Bx��N  "          Aff�.{A\)���W
=B֮�.{A=q@z�Ak�B��f                                    Bx����  �          A�׿���AG�?�p�A;�
B�{����@�=q@vffA׮B���                                    Bx����  T          A�����
@�=q@"�\A�Q�Bƙ����
@��H@�=qB{B�33                                    Bx���@  g          A�R����@�33?�Q�A�B�\����@�\)@Q�A���B���                                    Bx����  
y          A�Ϳ�RA33@\)As�B����R@�G�@�(�A�p�B�                                    Bx��Ȍ  
�          A
�H>ǮA (�@J�HA���B�p�>Ǯ@�33@�\)BQ�B�z�                                    Bx���2  �          A
{��A�@*=qA���B�\��@�G�@�Q�B��B�8R                                    Bx����  T          A��!�A?k�@���Bծ�!�@�G�@J=qA��HB���                                    Bx���~  T          Aff�(��A�?L��@�G�B��(��A�R@I��A�G�B��H                                    Bx��$  �          A���1G�A(�?�ffA�B���1G�@��H@`��A���B�#�                                    Bx���  T          A(��G�A�?�\)@��
B�W
�G�@�  @W�A�33BԨ�                                    Bx�� p  "          A��
=AG�@�\As�Bԙ��
=@���@�p�A�G�Bؙ�                                    Bx��/  T          A
=�G�A (�@!G�A��B�B��G�@�Q�@��
BffB�{                                    Bx��=�  �          A�\)A�
?��@�Q�Bя\�\)@�(�@U�A���Bӽq                                    Bx��Lb  �          A
=��p�A?���A��B�p���p�A�@�G�A�p�B�k�                                    Bx��[  �          Aff��\)A{@5A��HB�Q쿏\)@�Q�@�B�RB�\)                                    Bx��i�  �          A=q��p�A	�@G
=A�p�B�𤾽p�@�z�@�33B�\B��q                                    Bx��xT  �          A�\�c�
Az�@H��A�
=B�녿c�
@��H@��
B�B��
                                    Bx����  �          A녿^�RA�
@  Af=qB�ff�^�R@���@��\A�\B���                                    Bx����  �          A�׿���A
=>�@9��B��ῌ��A�@:=qA�ffB�                                    Bx���F  �          A�Ϳ�A��?�ff@�(�B��ÿ�@�p�@W�A�z�B���                                    Bx����  �          AQ��33A	p�?\)@h��Bɳ3��33A��@:=qA�33B���                                    Bx����  �          A33��{A�
>#�
?��
B�
=��{A@%�A�  B��                                    Bx���8  T          A\)���A  ��\)���B�LͿ��A�R@ffAt(�B�33                                    Bx����  �          A{��{A(�>�  ?�\)BŞ���{A��@*�HA�Q�B�z�                                    Bx���  �          A��  A\)���>�RBǀ ��  A�
?�p�AN�RB���                                    Bx���*  T          A�Ϳ��Az�Y����G�B�
=���A�H?ǮA#�B�Q�                                    Bx��
�  T          A����A�H�u�\B�W
��A
=q@��Aj=qB�W
                                    Bx��v  �          A��:�HA������w�B�uÿ:�HA
�\?��AD  B���                                    Bx��(  T          A���33A�?�(�A  BÀ ��33@�
=@qG�A�=qB���                                    Bx��6�  
�          Az�8Q�A�������HB�uÿ8Q�A\)@(�Ag33B�                                    Bx��Eh            A(��h��A33��\)��ffB�  �h��A�@
=A{
=B�u�                                    Bx��T  s          A��L��A\)�����	��B����L��A33@�Aap�B��                                    Bx��b�  T          A33��G�A
�R��
=�2�\B��;�G�A
=@G�AW33B���                                    Bx��qZ  �          A  ��A
=��
=�1G�B�\)��A\)@�AW�B��=                                    Bx���   �          Ap�?0��A  >��@*=qB�
=?0��Az�@6ffA�33B��=                                    Bx����  �          A��Q�AG����
�߮B�B��Q�A��?���A�
B�W
                                    Bx���L  �          A�
��p�A  ������\B�  ��p�A(�?�@�\)B���                                    Bx����  T          A���Az�>��@C33B��=��A��@:�HA�
=B���                                    Bx����  �          A
�\���A
=q>�ff@=p�B�
=���A�\@7
=A�ffB�(�                                    Bx���>  �          Az����A(�>�\)?�z�B�aH����Ap�@*�HA�  B��                                    Bx����  �          @�p���@���>�\)@33B���@�z�@p�A��B��                                    Bx���  �          @���@���>aG�?��BŸR��@��@�A��\Bƞ�                                    Bx���0  g          @�׿�=q@�\�\)����Bș���=q@�=q?�
=Apz�B�W
                                    Bx���            @�G�����@�=#�
>��
Bƅ����@�G�@�A�{B�Q�                                    Bx��|  
�          @��H��G�@�
=�Ǯ�H��B�p���G�@�G�?˅ANffB�
=                                    Bx��!"  g          @��H�أ�?��R�У��O�
C#���أ�@���  ��p�C ��                                    Bx��/�  
�          @�z����R�!G���
=��33C:  ���R�L�������RC4�                                     Bx��>n  "          @����p�?fff��p��J�\C*����p�?��H�n{��C'^�                                    Bx��M  T          @�����H?�\)��  ��C#�)���H?�=q�   ���\C!�
                                    Bx��[�  �          @��
��녾W
=�k��{C5�R��녾���\)�+�C5B�                                    Bx��j`  �          @ƸR��{���
>8Q�?�C6�R��{��Q�=��
?B�\C7Q�                                    Bx��y  �          @ʏ\�ə�>k�>�p�@Tz�C1���ə�=�>�
=@q�C2�                                    Bx����  �          @��˅>���?333@�G�C1L��˅=�Q�?B�\@ٙ�C333                                    Bx���R  �          @�=q��
=?n{?(�@�33C,���
=?8Q�?W
=@�(�C-��                                    Bx����  �          @�G���\)�8Q�?Tz�@ᙚC5�=��\)���?@  @�=qC7}q                                    Bx����  "          @����\)>8Q�>�=q@   C2����\)=���>��R@33C3G�                                    Bx���D  �          @����G�?\)=��
?��C/����G�?�\>u?޸RC0@                                     Bx����  �          @�\)���R>�녾��
�z�C1����R>��H�B�\����C0}q                                    Bx��ߐ  "          A ��� z�?�;#�
��z�C0
� z�?�        C/�                                    Bx���6  �          @�\)��ff?z���]p�C/�
��ff?0�׾����Q�C/�                                    Bx����  �          A
=��?s33����QG�C-Q���?���8Q쿝p�C,��                                    Bx���  
�          A
=�{?(��@  ���C/���{?L�Ϳ
=q�r�\C.\)                                    Bx��(  �          @����(�>W
=��Q��p�C2p���(�?
=q��=q� Q�C/��                                    Bx��(�  
�          @�  �Ǯ�@  ����i��C:ٚ�Ǯ��  ���
�~�HC6J=                                    Bx��7t  �          @�G����H�+�������C:O\���H    �ff���C4                                      Bx��F  �          @���ۅ>8Q쿹���?33C2���ۅ?zΎ��0Q�C/!H                                    Bx��T�  T          @ᙚ��\)?W
=�k���\)C-&f��\)?^�R<��
>#�
C,�f                                    Bx��cf  �          @�=q���
?���>��
@\)C'W
���
?�Q�?B�\@�  C(��                                    Bx��r  T          @��H��@{?#�
@�\)C!�H��?�Q�?�  A#�C#��                                    Bx����  �          @�z��\?��
���
�8Q�C#���\?��H>��H@�{C$O\                                    Bx���X  
�          @�  ��z�@��>�p�@A�C!����z�@   ?}p�A��C#8R                                    Bx����  �          @ᙚ�׮@ ��>���@)��C#^��׮?�=q?c�
@�Q�C$�                                    Bx����  �          @�=q����@
�H>���@*�HC!������?��R?p��@�{C#^�                                    Bx���J  T          @�ff��ff?��>�p�@0��C%�
��ff?�Q�?fff@�C'33                                    Bx����  "          A(����@?(��@��\C#u����@33?�ffA��C%ff                                    Bx��ؖ            A����?��
?!G�@��C'�f��?\?�\)@��RC)ff                                    Bx���<  �          A33��R@G�?E�@��C&����R?��H?��A33C(�                                     Bx����  �          A	G����?��?��@�33C'�����?�z�?��
A$��C*Y�                                    Bx���  �          A���@�
?�33@�{C$�=��?��?�G�A7�C'�                                    Bx��.  �          A��33?У�?��HA=qC)h��33?��?�\)A@z�C,�\                                    Bx��!�  �          A���?�ff?��
A\)C(����?��
?�p�AO33C+u�                                    Bx��0z  �          A�����@Vff?E�@�p�C
=���@>�R?ٙ�A.�RC :�                                    Bx��?   �          A����H@z�H?z�@j=qCaH��H@e?�Q�A+
=CO\                                    Bx��M�  �          A=q���\?�  ?���A1��C)0����\?z�H?�
=A[�
C,޸                                    Bx��\l  �          A
=����@G�?Y��@�  C#������?�
=?�(�A&�\C&\                                    Bx��k  �          A(�� ��@}p�?�@W�Cٚ� ��@h��?�33A(��C�R                                    Bx��y�  �          A�����@e?W
=@���C�)���@L��?�A:�RC�                                    Bx���^  �          A=q�   @�=q?W
=@�G�C���   @x��@33AO�
C\                                    Bx���  �          A�\���\@��?z�H@��C=q���\@�p�@�\Ag\)C�                                    Bx����  T          A����@�Q�?��
A ��C{����@�{@&ffA��RCE                                    Bx���P  �          Az����
@�?�  @��Cn���
@�{@�
AnffC.                                    Bx����  �          A����
=@g
=?J=q@�C����
=@N�R?�ffA=��C�                                    Bx��ќ  
�          A
=����@l(�?�G�@ָRC������@O\)@�\AY�Cn                                    Bx���B  �          A���z�@ ��?���@�z�C&���z�?�{?У�A*ffC)Q�                                    Bx����  T          AQ����?�{?
=q@a�C'���?�\)?�ff@���C)E                                    Bx����  �          A=q�G�?Y��?��@x��C.��G�?#�
?Q�@��C/޸                                    Bx��4  �          A����?!G�?Q�@��C/����>�Q�?xQ�@�=qC1��                                    Bx���  �          A���
?333?h��@��C/h���
>��?���@߮C1Q�                                    Bx��)�  �          A(��
�R?O\)?u@ȣ�C.���
�R?   ?�33@�C0��                                    Bx��8&  �          AG��Q�?z�?aG�@�ffC05��Q�>���?�G�@љ�C2                                    Bx��F�  �          Ap����>L��?W
=@�p�C2����ͽL��?\(�@�=qC4W
                                    Bx��Ur  �          A�\��>.{?Q�@�  C2���녽�\)?Tz�@��HC4z�                                    Bx��d  �          A
=��R��>��H@HQ�C4�
��R��=q>�(�@0  C5��                                    Bx��r�  �          A�\���?z�?z�@n�RC0@ ���>Ǯ?8Q�@��C1}q                                    Bx���d  �          A=q�
ff?���?�{A��C)���
ff?�{?�  A6{C,��                                    Bx���
  �          A��\)?�  ?���A�C(���\)?��
?�=qA;�C+�f                                    Bx����  �          A{��@%�?�33AG�C"�H��@�
@AY��C&(�                                    Bx���V  �          A(�����@`��?�\A;\)C� ����@6ff@-p�A���C�                                    Bx����  �          AQ���33@y��?��HAN�\C�\��33@J�H@@  A�ffC\)                                    Bx��ʢ  �          AG����@w�@�
AW�
C33���@G
=@EA�  C��                                    Bx���H  �          A�����@e@�A_
=C
���@5�@Dz�A�\)C�                                    Bx����  �          A(���z�@j=q@  An�\CaH��z�@6ff@N{A�Q�C�                                    Bx����  �          AQ����H@k�@
=AzffC!H���H@5@U�A��\C}q                                    Bx��:  �          A���  @��@p�A�p�CxR��  @N{@b�\A�ffC�                                    Bx���  �          A
�H���
@�p�@9��A���C:����
@X��@��A�p�C��                                    Bx��"�  �          A
�R���H@�33@]p�A�Q�C\���H@xQ�@�G�B33C��                                    Bx��1,  �          A	p���=q@��
@aG�AÅC
����=q@xQ�@�33B
=CQ�                                    Bx��?�  �          A
=q��
=@��\@�G�A��C	���
=@g�@��\B!z�C��                                    Bx��Nx  �          A	�����@Fff@��
BG�C������?Ǯ@��B"�RC&�                                    Bx��]  �          AQ���33@dz�@�
A|��C
��33@0  @P  A�C�                                     Bx��k�  �          A����Q�@k�?�  A<��C�H��Q�@AG�@/\)A���C�                                    Bx��zj  �          A
=��?O\)?�ffAz�C.z���>���?�p�A   C1E                                    Bx���  �          A
�R��Ϳ�>u?�{C=k���Ϳ�
=�8Q쿗
=C=}q                                    Bx����  �          A
�\��ÿ�G�>Ǯ@%C<Y���ÿ���<#�
=��
C<�q                                    Bx���\  �          A��
�R���
>.{?��C:�
�R���
���\(�C:��                                    Bx���  T          A�R�p���p�<�>8Q�C;�f�p���
=�����
=C;��                                    Bx��è  �          A�(���\)<��
>��C<���(����þ�p���HC<�{                                    Bx���N  �          A���Q쿜(�=�G�?5C;��Q쿙����  ����C;�=                                    Bx����  �          A��Ϳ��
=���?&ffC:����Ϳ�G��L�Ϳ��C:��                                    Bx���  �          A��
�H�\(��\�{C9���
�H�8Q�(���  C8�R                                    Bx���@  �          A��
�\�k����^{C:
=�
�\�:�H�E����C8Ǯ                                    Bx���  �          A���33�n{�5��  C:@ �33�.{�p����33C8�)                                    Bx���  A          A
=q�	��aG������Q�C9ٚ�	��+��Tz�����C8s3                                    Bx��*2  T          A	��ÿL�ͽ�\)��G�C9W
��ÿ@  ��z���HC8�q                                    Bx��8�  
�          A���Q�>���>��@QG�C1�3�Q�>8Q�?��@s33C2�q                                    Bx��G~  �          AQ��ff?s33?�  @�C-\)�ff?�R?�p�A��C/��                                    Bx��V$  T          Az��>��H?\A)G�C0���=#�
?˅A2{C3�                                     Bx��d�  "          A  ���R?�Q�?��AH��C+u����R?!G�@�
Ag\)C/}q                                    Bx��sp  "          A��G�?G�?�=q@��C.z��G�>�G�?�G�AG�C0�f                                    Bx���  g          A��p�?(��?�z�A��C/^��p�>�z�?�ffAp�C1��                                    Bx����  �          A��ff?n{?�@n{C-� �ff?=p�?G�@�C.�
                                    Bx���b  �          A���>�33�u���C1����>�33=L��>��RC1��                                    Bx���  5          A��33��p��B�\���C6���33������z���C6)                                    Bx����  s          A���R��{�:�H��ffC=�����R��������Q�C;��                                    Bx���T  
�          Az�� Q������H��\C>�
� Q쿏\)�����2�\C;�R                                    Bx����  
�          A�H��ff���H���R��C>h���ff�����{�5p�C;xR                                    Bx���  
�          A�R��{���R�����\)C>�H��{��=q��=q�2ffC;                                    Bx���F  T          A����H��\)����Q�C?�����H����  �G\)C<z�                                    Bx���  
m          A  ��ff��zῼ(��$z�C?Ǯ��ff��z����S�C<T{                                    Bx���  �          A(�� z῵�����33C>  � z�z�H���:�HC:�                                    Bx��#8  "          A�H��ff���ÿ�33��C=u���ff�^�R��(��B�\C:8R                                    Bx��1�  T          A�
�{��=q������C5��{=�\)���33C3��                                    Bx��@�  �          Az�������Y����z�C5Ǯ��    �c�
����C4�                                    Bx��O*  T          A(���>�Q�333���C1xR��?�Ϳ��{�C00�                                    Bx��]�  T          A�
�33>Ǯ�0������C1B��33?녿���tz�C0                                      Bx��lv  g          A(���
?�\�u��Q�C0p���
?\)��Q���C0�                                    Bx��{  �          A����
?^�R�W
=��Q�C-�3��
?fff=L��>�33C-�=                                    Bx����  �          Az��(�?녾aG��\C0{�(�?���L�;�{C/�\                                    Bx���h  "          AG��\)���
����{C4���\)>��ÿ�{�{C1��                                    Bx���  �          A���ý�Q����2ffC4�����>\�˅�-C1^�                                    Bx����  g          A��=q�L�Ϳ��H�33C4W
�=q>�������ffC1�R                                    Bx���Z  A          A  �
=<#�
�u��  C3�R�
=>�\)�k���
=C2)                                    Bx���   T          A(���H����z����C4�f��H>L�Ϳ�33���C2��                                    Bx���  
�          A�\�����
��Q��=qC4�\��>����z����RC233                                    Bx���L  T          AG����aG���(���HC5���>\)��p��(�C3�                                    Bx����  "          A���ff>��ÿ��H�"=qC1�3�ff?:�H�����C.��                                    Bx���  "          A����R?!G������(�C/����R?fff�W
=��=qC-�                                    Bx��>  T          A���(�@(��c�
�ǮC$xR��(�@ff�k��У�C#^�                                    Bx��*�  
�          A\)����@G���G���\C%�����@{�����1G�C$J=                                    Bx��9�  �          A����@
=���R�
=C$����@Q�
=���
C##�                                    Bx��H0  �          A33��
=?��
��\)���C*�f��
=?�{�s33��z�C(��                                    Bx��V�  "          A�H���?�녿��R�
=C(G����?��8Q�����C&T{                                    Bx��e|  �          A33��33@33��{��=qC%Y���33@녾��H�Y��C#Ǯ                                    Bx��t"  T          A
=��=q@G��W
=��(�C#�\��=q@�H�#�
����C"�
                                    Bx����  
�          A
=����@�
�n{�У�C#� ����@�R�u��z�C"^�                                    Bx���n  
;          A����@Q쿅���\)C#\���@$zᾣ�
��RC!                                    Bx���  �          A33��  @�ÿ�Q����C"�H��  @'�����QG�C!G�                                    Bx����  �          A���G�@녿�����RC#���G�@%��.{��Q�C!��                                    Bx���`  �          A����p�?�Q��{�333C&:���p�@z῀  ���C#��                                    Bx���  T          AG���
=?�\)��=q�/33C&�\��
=@\)�}p����HC$O\                                    Bx��ڬ  T          A{��@�\��\�C
=C%����@�Ϳ�\)���RC"��                                    Bx���R  
�          AG���z�@녿��H�=p�C%����z�@��������
C"�                                    Bx����  �          A�����
@G���
=�:=qC%�����
@=q�����{C"�q                                    Bx���  �          A�
���@C�
������33C�3���@P�׾aG��\C��                                    Bx��D  "          A  ���@L(��&ff����C&f���@O\)>�=q?��C��                                    Bx��#�  
�          A���
=@fff=#�
>���Cz���
=@\��?�ff@�Cz�                                    Bx��2�  �          AG���p�@����\)����C���p�@�=q?fff@��C�\                                    Bx��A6  �          AG�����@z=q�333��  C������@|��>�
=@?\)C��                                    Bx��O�  
�          Ap���@l�Ϳ�����=qC����@w
=���
���C�                                    Bx��^�  �          AG����H@j=q���
���C�3���H@w��L�Ϳ�Q�C^�                                    Bx��m(  g          AG���Q�@n{����/�C���Q�@�Q��G��FffC=q                                    Bx��{�  �          A�\���@�
�����5p�C#@ ���@*=q�fff���C �{                                    Bx���t  �          A���(�@33�\�,��C#8R��(�@(Q�Q�����C ��                                    Bx���  �          A�陚@@�׿����O�
C���陚@Y���h����{C                                    