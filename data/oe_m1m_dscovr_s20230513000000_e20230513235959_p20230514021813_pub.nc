CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230513000000_e20230513235959_p20230514021813_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-14T02:18:13.151Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-13T00:00:00.000Z   time_coverage_end         2023-05-13T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�&��  �          @]p��Fff�^�R��
=���
CC�3�Fff�!G��=p��QC?ff                                    Bx�&�&  �          @a��J�H���>�Q�@�p�C>���J�H�0��=�Q�?��C@\)                                    Bx�&��  �          @n{�9��>�\)@  B�RC.p��9���(��@
�HBp�C@�{                                    Bx�&�r  "          @���	��>�(�@l��Bmz�C(�3�	����\)@c33B^�COz�                                    Bx�&�  �          @�Q����Q�@��HB�33C6�Ϳ�����@{�B_{C_�q                                    Bx�&̾  �          @��ÿ�\)��(�@^{Bt  C@�R��\)��@?\)BC�C_�R                                    Bx�&�d  �          @�ff�#�
���
@Tz�BQ�C4k��#�
��\)@A�B8�HCP.                                    Bx�&�
  "          @\)�'
=<�@>{BB�C3J=�'
=��
=@.{B.G�CLG�                                    Bx�&��  �          @!G���
=>W
=?�  BJ{C+���
=��?�
=B?��CH�                                    Bx�'V  
�          @@  �
=?\)?�\)B"=qC%��
=��  ?�Q�B)ffC:�H                                    Bx�'�  �          @|���{@�@p�B�RC��{?��\@>�RBEffC0�                                    Bx�'$�  T          @��R��H@*�H@(�A��HC#���H?�ff@EB<ffCQ�                                    Bx�'3H  �          @4z��?��?   A*=qCc׿�?\?�(�A�Cz�                                    Bx�'A�  �          ?У׿p��?�G��u��B�k��p��?�
=>�ffA��C �{                                    Bx�'P�  �          @,�Ϳ\?˅?�ffA���C���\?��
?�{B%{C��                                    Bx�'_:  �          @[���\?\@�B'CO\��\>�@,��BQ�C'(�                                    Bx�'m�  T          @q���R?��@�HB$��C�3��R?�@7�BM��C&޸                                    Bx�'|�  
�          @�33�G�?У�@>�RB:��CQ��G�>�z�@XQ�B_p�C,�3                                    Bx�'�,  �          @�  ��
=?s33@^�RBs�
Cz��
=��@dz�B~CE��                                    Bx�'��  �          @������?J=q@vffB��HCk����ÿTz�@uB�=qCO��                                    Bx�'�x  "          @��׿���?W
=@�ffB��
CJ=���Ϳ�p�@��
B���CYn                                    Bx�'�  
�          @�=q���?&ff@���B��3C&f��녿��R@�p�B�=qC]�q                                    Bx�'��  �          @����?�(�@�G�Bh�C�)�녾#�
@��
B�L�C8c�                                    Bx�'�j  "          @��׿�{?�  @Tz�BQCٚ��{>��@o\)B}G�C+�q                                    Bx�'�  T          @y����\?�(�@&ffB)�C  ��\?B�\@J�HB^{C�
                                    Bx�'�  �          @j�H�ٙ�?�ff@%B9p�CO\�ٙ�?(�@FffBoQ�C \)                                    Bx�( \  �          @u���p�?˅@6ffBA(�CJ=��p�>��R@P  Bi33C+�                                    Bx�(  �          @���0��@�\@I��B%{CO\�0��?W
=@s�
BSQ�C#+�                                    Bx�(�  T          @�G��AG�@�@4z�B�C0��AG�?���@c�
B?z�C 
=                                    Bx�(,N  �          @��H�#33?˅@��B^��C�q�#33���R@�Bu
=C:��                                    Bx�(:�  T          @�(��%?�  @��\BrC�f�%�B�\@�\)B=qCDW
                                    Bx�(I�  �          @��H�
=?}p�@�33B��C��
=���@���B�ffCTn                                    Bx�(X@  �          @�=q��Q�?O\)@���B�z�C:��Q��z�@�33B��C`�)                                    Bx�(f�  T          @�(����
?�p�@�
=B���CW
���
��z�@��B�ffCU
                                    Bx�(u�  �          @���?\)@�
@�ffBD�\Cz��?\)>W
=@�p�Be(�C/�3                                    Bx�(�2  �          @��R�,��?�{@��BT�C}q�,�ͼ�@��Bq(�C4��                                    Bx�(��  �          @�p��   ?�@�z�B�B�C���   �aG�@��RB�ffCK��                                    Bx�(�~  �          @�z���?�@�33B�p�C"c׿�녿�z�@��
B��CaW
                                    Bx�(�$  �          @��׿���?aG�@�33B�k�C�{��������@�Q�B��
CV�                                    Bx�(��  "          @��\��?=p�@��B�ǮC�=����  @��\B�\)C[��                                    Bx�(�p  �          @�G���=q?G�@�{B��
C����=q��Q�@���B��=CZO\                                    Bx�(�  "          @�녿�?k�@��B���C� ����ff@��\B��HCV!H                                    Bx�(�  
�          @��\���R?��@��
B�aHC����R��33@�33B�p�CR�                                    Bx�(�b  T          @���{?�(�@���B���CJ=�{�z�H@��\B�{CK�q                                    Bx�)  �          @�ff�%�?��R@�
=Bu(�CO\�%��n{@�G�Bzp�CG��                                    Bx�)�  �          @��\�{?���@�
=B~z�C:��{�Y��@�=qB�ǮCH�)                                    Bx�)%T  �          @�Q��
�H?��H@�
=B�.CǮ�
�H��ff@�Q�B�CM�\                                    Bx�)3�  �          @�
=����?���@���B���Cp�������=q@�G�B�8RCP�R                                    Bx�)B�  �          @�{��?�=q@��B�k�C�������@���B�.CUs3                                    Bx�)QF  T          @�z���H?xQ�@�=qB��{C�����H��33@�\)B��qC[@                                     Bx�)_�  
�          @�=q���R?���@��
B�ǮC�f���R���H@��
B��RCS8R                                    Bx�)n�  T          @ȣ׿�
=?���@�  B��=C����
=����@�ffB�W
C_^�                                    Bx�)}8  �          @�G�����?���@\B��fCE���Ϳ���@�G�B�ǮCf��                                    Bx�)��  �          @��H���?�z�@�
=B���C�������\@�ffB�p�CWaH                                    Bx�)��  �          @θR���?�{@�
=B�#�CB���Ϳ���@�p�B�(�CS                                      Bx�)�*  "          @�  �p�?��H@��B�z�CT{�p����R@��B�.CQQ�                                    Bx�)��  �          @�Q��p�?��@�\)B��CW
�p����@���B��CO8R                                    Bx�)�v  T          @�� ��?�(�@�B�aHC� �׿u@���B�p�CM�{                                    Bx�)�  T          @�(��p�?�G�@�G�B�G�C���p��^�R@�p�B�.CIu�                                    Bx�)��  T          @�\)� ��?�p�@���B|�
C�=� �׿c�
@���B�L�CG�{                                    Bx�)�h  �          @�
=��ff?�
=@���B�C�ÿ�ff���\@���B��
CW&f                                    Bx�*  T          @��
=q?c�
@��HB���B�k��
=q��ff@�ffB��Cz�                                    Bx�*�  T          @ƸR��?��@���B�  B�(�����{@�=qB�#�CtT{                                    Bx�*Z  T          @�{�.{?�
=@�G�B��{B��.{��G�@���B�L�Cq                                    Bx�*-   "          @ƸR��ff?��@\B�z�B�=q��ff����@���B���C{�
                                    Bx�*;�  "          @�33�\)?k�@���B�z�B�W
�\)���R@��B��C�S3                                    Bx�*JL  
�          @ȣ׿�33?p��@���B��C!H��33��(�@�B�#�CbQ�                                    Bx�*X�  "          @�(���
=?s33@�B�u�C(���
=��
=@��HB��HCfz�                                    Bx�*g�  �          @\��33?�33@���B�ffC���33��Q�@�G�B��
C\\)                                    Bx�*v>  �          @�\)����?��\@��
B�\CaH���Ϳ�  @�B�Q�CS��                                    Bx�*��  T          @�z��G�?��@�G�B�G�C
��G��n{@��B��CS�)                                    Bx�*��  �          @������?���@��\B�G�C	�������n{@���B�#�CT�                                    Bx�*�0  �          @��R����?��@�B��
C�ÿ����s33@�Q�B�L�CZ^�                                    Bx�*��  �          @�ff��  ?�=q@��HB�u�C
z��  �k�@�p�B�p�CS��                                    Bx�*�|  �          @�33����?��@�
=B���C�׿��Ϳfff@���B��=CQG�                                    Bx�*�"  �          @�p���p�?���@�
=B�#�C�׿�p��B�\@��B��
CK��                                    Bx�*��  �          @��R�	��?�p�@���Byz�C+��	����
=@���B�ǮC?+�                                    Bx�*�n  �          @�z��
=?���@��\Bnp�Cff�
=��=q@�z�B�k�C:u�                                    Bx�*�  �          @�{�  ?��H@�
=Bv33C��  ��
=@�\)B���C>�                                     Bx�+�  �          @����(�?�{@�BxffC�R�(���Q�@�\)B��{C=G�                                    Bx�+`  �          @��˅?�{@�  B��C��˅�333@�B�8RCK�
                                    Bx�+&  �          @θR���
?�  @�ffB�C�
���
���H@ƸRB�\)C_O\                                    Bx�+4�  "          @��Ϳ���?�{@���B�� C�3���Ϳ��@\B�#�CT�H                                    Bx�+CR  �          @�  �
=q?Ǯ@���B�� C\�
=q�5@�=qB�ffCF
                                    Bx�+Q�  �          @���?��R@��Bz\)C����5@�z�B��3CD\)                                    Bx�+`�  �          @�\)�1�?ٙ�@��Bi��C�\�1녾�ff@��B|��C=�                                    Bx�+oD  �          @�{��?���@�G�B33C=q���Y��@�z�B��CG�                                     Bx�+}�  �          @�  � ��?���@���B�{C�� �׿��@�=qB�.CPp�                                    Bx�+��  "          @ȣ����?�(�@��B��C���׿G�@���B�33CG�                                    Bx�+�6  T          @��
�*�H?Ǯ@�=qBs�C���*�H�(��@��B���CA��                                    Bx�+��  �          @��H�/\)?У�@�
=Bn��C:��/\)���@�B(�C?k�                                    Bx�+��  �          @��H�2�\?�G�@�\)Bo�HC���2�\�+�@���B|\)CA�                                     Bx�+�(  �          @�33�C33?�z�@��Bc{Cff�C33��@�G�Bs�\C<�                                    Bx�+��  �          @ʏ\�:�H?�@��
Bg��CO\�:�H���@�33Bxz�C=33                                    Bx�+�t  �          @ə��{?��
@��HBz=qC8R�{�.{@�  B�
=CC�                                     Bx�+�  �          @�  ��?�@��HB��
C��������@�33B��CQ��                                    Bx�,�  �          @�{�{?���@���B~�C{�{�xQ�@��HB�B�CIaH                                    Bx�,f  �          @����   ?�33@���Bvz�C�R�   �0��@�B�{CC\)                                    Bx�,  T          @����0��?�=q@��BhffCJ=�0�׾�ff@��\By(�C=8R                                    Bx�,-�  T          @�=q�8��?�@�=qBb�C��8�þ�33@�=qBup�C:ٚ                                    Bx�,<X  T          @Å�>�R?��H@���B_{C
�>�R��z�@�=qBr��C9��                                    Bx�,J�  �          @Å�B�\?޸R@�Q�B\Q�C5��B�\��  @���BpffC8Ǯ                                    Bx�,Y�  T          @�33�>�R?ٙ�@���B_33Cff�>�R���R@�=qBrG�C9�
                                    Bx�,hJ  T          @����J�H?���@�Q�BZ�RC(��J�H��p�@�  Bj�
C:��                                    Bx�,v�  �          @�{�P  ?�G�@�G�BZ=qC#��P  ���@��Bg�\C<8R                                    Bx�,��  �          @ƸR�h��?��
@�ffBD��C�)�h�ý��
@���BXp�C5T{                                    Bx�,�<  "          @�Q��l��@ ��@�z�B?(�Ck��l��>��@��BW33C1�H                                    Bx�,��  �          @��
�@  ?��@�p�Bf��C^��@  �0��@�G�Bo33C@��                                    Bx�,��  �          @�\)�QG�?��@��
B]C#��QG��+�@��Be��C?�\                                    Bx�,�.  �          @�=q�]p�?�  @�
=BPQ�C33�]p��W
=@���Bb��C7xR                                    Bx�,��  T          @��H�N�R?Ǯ@��B^Q�C0��N�R��@�{Bl33C<^�                                    Bx�,�z  �          @��H�QG�?���@��B_�RC!H�QG��&ff@�(�Bi  C?0�                                    Bx�,�   �          @�=q�G�?�ff@���Ba��C���G����H@�
=Bo�C<��                                    Bx�,��  
�          @��H�u?�p�@�\)BAz�C��u��@�G�BRC5�                                    Bx�-	l  �          @���y��?��@��
B<Q�CY��y�����
@��RBO{C4\)                                    Bx�-  �          @�\)��33?�(�@��B,33C\)��33>�\)@�p�BB�RC0!H                                    Bx�-&�  �          @ƸR���@�\@8��A�(�C^����?���@a�B
�\C'ٚ                                    Bx�-5^  �          @Ǯ��@�@l(�BG�C����?\(�@���B.��C(�R                                    Bx�-D  "          @ȣ��fff@�@���B?\)C�q�fff>�{@��
B[33C.�{                                    Bx�-R�  
�          @������@
=@q�BC�����?Q�@�(�B5��C)�                                    Bx�-aP  
�          @�
=��(�@�
@o\)BG�C5���(�?J=q@�=qB0�C)�                                     Bx�-o�  T          @ʏ\��p�@G�@�{B%��Ck���p�?��@�\)BAG�C+޸                                    Bx�-~�  �          @�z���{@  @�Q�B'=qC�q��{?\)@�G�BB{C,ff                                    Bx�-�B  �          @�z���=q@
�H@��B.�HC�R��=q>�(�@���BHffC.
=                                    Bx�-��  
�          @�z��fff@33@��BF�Cff�fff>��@���B^p�C1��                                    Bx�-��  �          @����l(�@�
@��BB�C�H�l(�>B�\@�\)BZ�
C1�                                    Bx�-�4  
�          @�ff�k�?�(�@��BFffC�
�k�=�\)@�G�B\��C2��                                    Bx�-��  <          @�p��g
=?�z�@��RBJ(�C.�g
=�#�
@�=qB_33C4
                                    Bx�-ր  "          @�G�����?��R@���B%�C  ����>�p�@��HB;��C/{                                    Bx�-�&  T          @������?���@xQ�BQ�C8R����>�{@���B0z�C/�f                                    Bx�-��  �          @ȣ�����?�=q@�ffB(�Cٚ����>W
=@�=qB;p�C1=q                                    Bx�.r  
�          @�33���\?���@�Q�B5  C����\=�Q�@��BG�
C2�\                                    Bx�.  �          @ȣ����
?�{@�{B4z�C�f���
���
@�
=BC��C5!H                                    Bx�.�  �          @�{����?���@�(�B4{C!�����;�\)@��\B?  C7�=                                    Bx�..d  T          @���z�H?��\@��B?�C"�z�H��(�@��RBHG�C:Q�                                    Bx�.=
  �          @ə��<��?޸R@���Bd{C��<�;�  @���Bw��C8޸                                    Bx�.K�  
�          @�Q��J=q?ٙ�@��
B[�C�3�J=q�k�@�z�BnG�C8�                                    Bx�.ZV  
�          @ȣ��^�R?�=q@�
=BQ��C���^�R��\)@��RB`�\C8��                                    Bx�.h�  
�          @�\)�^{?���@�p�BP�CE�^{��  @�p�B`33C8
=                                    Bx�.w�  T          @�Q��U?��H@�p�BO=qC�f�U=�Q�@���Bf��C2n                                    Bx�.�H  "          @�  �R�\@G�@��BO{Cu��R�\>#�
@���Bh�C10�                                    Bx�.��  �          @ƸR�<(�?У�@�
=Bd�C���<(����R@��RBvffC:�                                    Bx�.��  
�          @�(��{?��@�By��C���{���@�=qB�  CA�                                    Bx�.�:  �          @��
�.�R?�Q�@�G�Bn�HC+��.�R��\@��RB|�C>��                                    Bx�.��  "          @�ff�2�\?�
=@��Bn��C�)�2�\�
=q@���B{�C>�R                                    Bx�.φ  
�          @��/\)?�
=@�z�Bu(�CY��/\)�!G�@�G�B�8RC@�q                                    Bx�.�,  �          @�Q�� ��?�\)@���Bz�RC5�� �׿   @�\)B���C?^�                                    Bx�.��  �          @У��.�R?\@�
=Bu��C���.�R��@�z�B�ffC?��                                    Bx�.�x  T          @����E�?�Q�@�  Be��C(��E�����@��Bv�RC:�                                    Bx�/
  
�          @�\)�HQ�?��@�z�Ba{CB��HQ�L��@�Btp�C7��                                    Bx�/�  �          @˅�.�R?���@���BpC�
�.�R��
=@�\)B���C<��                                    Bx�/'j  �          @�33�,(�?�(�@��Bt\)Cc��,(����@�
=B��C?}q                                    Bx�/6  T          @�p��7�?�@�Q�Bj��C�q�7����H@�Bw  C=��                                    Bx�/D�  �          @�p��9��?���@��Bi33Cff�9����ff@�p�Bvp�C<ٚ                                    Bx�/S\  �          @Å�)��?�Q�@��\Br
=Cn�)����@�  B�RC>aH                                    Bx�/b  T          @����)��?�p�@�33Bq�C�f�)����ff@���B�{C=�q                                    Bx�/p�  �          @����,(�?�{@��Br��C#��,(��\)@�  B}�HC?�
                                    Bx�/N  �          @��
�'�?�33@�33Bs�C޸�'���@�Q�B�G�C?E                                    Bx�/��  �          @\�*=q?��@���Brp�C5��*=q���@�{B}�C?�3                                    Bx�/��  �          @�����?���@�
=B��C�f�녿O\)@���B��RCG��                                    Bx�/�@  
�          @�z��>�R?�ff@��Ba��C��>�R��=q@�G�Bq�
C9+�                                    Bx�/��  �          @���XQ�?�G�@�=qBN��C}q�XQ�<#�
@�(�Bbp�C3Ǯ                                    Bx�/Ȍ  �          @���:�H?У�@��Bd{C�H�:�H�aG�@���Bv
=C8^�                                    Bx�/�2  �          @��
�G
=?�Q�@��BZ�C��G
=��G�@���Bm�RC6�                                    Bx�/��  �          @��A�?���@�z�Ba�C���A녾�=q@��
Br
=C9)                                    Bx�/�~  �          @�p��?\)?�ff@�p�Bc��C��?\)��z�@�z�Bs�C9�                                     Bx�0$  �          @��8��?��@��Bh
=C�f�8�þ��
@��RBw�
C:T{                                    Bx�0�  �          @ƸR���?���@�p�Bs�C޸�����@�
=B�W
C6                                    Bx�0 p  T          @�\)��R?ٙ�@��RBt�\C����R�u@��RB��)C9n                                    Bx�0/  "          @�  ��H?���@�ffBs{C�
��H��Q�@�Q�B�Q�C6�                                    Bx�0=�  "          @�Q��)��?���@�=qBh��C���)��=L��@�p�B���C2�)                                    Bx�0Lb  
Z          @�
=�333?��
@��\Bl\)Ch��333��33@�G�B{�HC;\                                    Bx�0[  T          @�p��@�@�  Bj��C	��>�z�@�B��C,�{                                    Bx�0i�  T          @�ff�/\)?�
=@�  BjQ�C���/\)�.{@�Q�B~33C7��                                    Bx�0xT  �          @�Q��;�?�  @�\)Bc�\C!H�;���Q�@���Bx�C5��                                    Bx�0��  T          @��H�Mp�?�
=@�ffB\ffCff�Mp��\)@��RBn  C6�{                                    Bx�0��  T          @��H�N{?˅@�\)B^{CǮ�N{�u@��RBm�\C8@                                     Bx�0�F  T          @˅�QG�?�Q�@�{BZ��C��QG���@�ffBl(�C633                                    Bx�0��  
�          @�=q�C33@�@���BZQ�CO\�C33>W
=@���Bt\)C/�q                                    Bx�0��  �          @ȣ��<(�@\)@��BW\)C���<(�>�G�@�Q�Bv��C+n                                    Bx�0�8  �          @����1�?�@��
Bb33Cc��1�>�@��RB{��C1ff                                    Bx�0��  T          @��H�(�?�  @�G�BqC=q�(����
@��\B�ǮC6                                      Bx�0�  T          @�=q���?�
=@�=qBt�
C  ��þ#�
@�=qB�k�C7�f                                    Bx�0�*  �          @�Q��$z�?޸R@���Bk�RC�)�$z�#�
@�B�k�C4��                                    Bx�1
�  �          @���"�\?��H@���Bt33C
�"�\��33@��B��HC;�f                                    Bx�1v  �          @�33�p�?�@�(�Bx��C�3�p����@���B��{C=}q                                    Bx�1(  T          @����p�?�\@�33Br(�CJ=�p����
@�z�B��fC5�                                    Bx�16�  �          @�z��!�?�z�@�Q�Bk�C�R�!�=���@��HB�8RC1�\                                    Bx�1Eh  
�          @�(��#33?��@��\BqQ�CL��#33�8Q�@�=qB���C8                                    Bx�1T  T          @��
��H?�\)@�ffB|
=C���H���@��HB��C?�                                    Bx�1b�  T          @��
� ��?�(�@�(�Bv=qC�� �׾�33@�=qB���C;�f                                    Bx�1qZ  "          @��
�\)?�33@��HBs(�C� �\)�#�
@��\B��)C7�                                    Bx�1�   �          @����)��?�  @��HBq{C� �)������@�G�B��C:n                                    Bx�1��  �          @���333?�(�@�p�Bi�CE�333��=q@��
By33C9k�                                    Bx�1�L  
�          @����7�?�Q�@�33Bg{C^��7���=q@�G�Bu�\C9h�                                    Bx�1��  "          @�  �9��?��R@���Bd(�Cٚ�9���L��@��Bs�HC7�R                                    Bx�1��  �          @�  �@  ?�(�@��B`�RC�f�@  �L��@�{BoC7�
                                    Bx�1�>  
�          @���Dz�?�{@�(�BZp�Cn�Dz�#�
@�z�BlffC4�                                    Bx�1��  T          @Å�C�
?У�@���B]��C��C�
�u@�G�Bo�
C5
                                    Bx�1�  �          @�33�C33?���@�=qBZ�C�3�C33�#�
@�=qBk�
C4��                                    Bx�1�0  �          @��
�Z=q?��@j=qB.=qCW
�Z=q?�@�G�BE��C+)                                    Bx�2�  
�          @�33�k�@{@J=qBG�C���k�?�{@l��B/��C#J=                                    Bx�2|  T          @���n�R@@C33B
�C��n�R?�G�@hQ�B*��C!h�                                    Bx�2!"  T          @��
�l(�@G�@K�B  Cp��l(�?�33@n�RB/�HC"�3                                    Bx�2/�  �          @���g�@(�@P��B�C�{�g�?�ff@r�\B4p�C#��                                    Bx�2>n  T          @�z��n�R@	��@P  B�C\�n�R?��\@p��B0�\C$�                                    Bx�2M  T          @�33�k�@�@N{B�RC^��k�?��@o\)B1�C#�f                                    Bx�2[�  �          @���hQ�@p�@P��B{C�H�hQ�?�=q@r�\B4(�C#k�                                    Bx�2j`  �          @����p��@�R@>{B�RC\)�p��?�Q�@`��B&��C"p�                                    Bx�2y  "          @��
�j=q@�@5B�C0��j=q?�Q�@XQ�B%Q�C"                                    Bx�2��  �          @�=q�S33?�@s�
BC{C+{�S33�0��@q�BA(�C?ٚ                                    Bx�2�R  
�          @���^�R?5@uB<�C(}q�^�R��@w�B>�HC<s3                                    Bx�2��  "          @����j=q?8Q�@p��B4��C(���j=q��@s33B7(�C;u�                                    Bx�2��  T          @����xQ�?^�R@^�RB$�RC'W
�xQ쾀  @e�B*z�C7��                                    Bx�2�D  "          @��u�?0��@[�B%�C)Ǯ�u�����@^{B(=qC9��                                    Bx�2��  �          @�33�Vff>���@u�BB��C.T{�Vff�\(�@o\)B=(�CBn                                    Bx�2ߐ  
�          @�(��\(�?z�@p��B<ffC*��\(��(�@p  B<{C=��                                    Bx�2�6  �          @�z��e?!G�@g�B3�C*  �e��\@h��B4ffC<�                                    Bx�2��  �          @��
�g
=>�@eB2\)C,���g
=�(��@c�
B0p�C>O\                                    Bx�3�  �          @�p��fff>�
=@k�B5z�C-aH�fff�8Q�@hQ�B2z�C?T{                                    Bx�3(  "          @�\)�aG�>���@vffB=��C.���aG��\(�@qG�B8�CA��                                    Bx�3(�  T          @��X��>���@y��BC�\C.�H�X�ÿfff@s�
B=p�CB�=                                    Bx�37t  �          @�(��b�\>u@l��B8�C0+��b�\�fff@fffB2ffCB8R                                    Bx�3F  "          @���l(�=���@]p�B,�C2c��l(��s33@U�B$�RCBh�                                    Bx�3T�  "          @��\�xQ�>\)@QG�B �C1��xQ�Y��@J=qB
=C@k�                                    Bx�3cf  �          @�Q��i��?   @XQ�B*\)C,��i���
=q@XQ�B*
=C<aH                                    Bx�3r  T          @���Z=q>�z�@hQ�B:�C/8R�Z=q�Q�@c33B5G�CA}q                                    Bx�3��  "          @�
=�:�H?xQ�@x��BN��C!���:�H��  @�  BWffC8�\                                    Bx�3�X  �          @�Q��>{?&ff@j=qBH��C'���>{��@k�BJ�C=�                                    Bx�3��  �          @�{���?#�
@~�RBh=qC%���ÿz�@~�RBi{CA��                                    Bx�3��  T          @����?
=@���B|Q�C$)��333@�  Bz�CF��                                    Bx�3�J  �          @�Q��*=q?�{@�G�BZ{CT{�*=q�\)@�{BfG�C7�                                    Bx�3��  �          @����*�H?�33@��HBZG�C�q�*�H��@�  Bg(�C6u�                                    Bx�3ؖ  �          @�33�4z�?Tz�@��B[{C#� �4z��(�@�
=B_�HC<�                                     Bx�3�<  T          @�Q��0  ?xQ�@��BYffC �\�0  ��=q@�p�Ba�HC9��                                    Bx�3��  "          @����+�?\(�@���B_Q�C"#��+�����@�
=Be(�C<h�                                    Bx�4�  
�          @��\�#�
?^�R@���BgffC!O\�#�
��G�@��Bm  C=�f                                    Bx�4.  �          @�=q�8��?aG�@�=qBUffC#��8�þ�33@���B[��C:�)                                    Bx�4!�  �          @����~�R?�G�@n{B(�C%���~�R��@vffB0�C5�                                    Bx�40z  T          @�����z�?:�H@g�B#
=C*���zᾸQ�@j�HB&
=C8�q                                    Bx�4?   
(          @�  ��
=?   @]p�B�C-:���
=��@]p�BQ�C;�                                    Bx�4M�  �          @����tz�?�G�@j�HB+�RC%:��tz����@s�
B3��C5�\                                    Bx�4\l  �          @�{�U�?0��@y��BC�C(8R�U���@{�BE��C;�                                    Bx�4k  �          @�Q��^�R?0��@vffB=Q�C(�q�^�R��ff@x��B?��C;\)                                    Bx�4y�  �          @���U�>�@��BJ��C,
�U��=p�@�=qBH(�C@�
                                    Bx�4�^  �          @���\(�?
=q@�=qBE��C+�\(��&ff@���BD��C>�                                    Bx�4�  �          @��
�e?(��@{�B<��C)���e�   @|��B>{C;�3                                    Bx�4��  �          @��W�?�R@xQ�BBffC)�=�W���@y��BCp�C<��                                    Bx�4�P  T          @�
=�W�?&ff@{�BCQ�C)��W��   @|��BD�
C<aH                                    Bx�4��  �          @��
�Z=q?(��@p��B=33C)��Z=q��G�@s33B?=qC;h�                                    Bx�4ќ  �          @�ff�a�?0��@p  B8�C(��a녾��@s33B;��C:��                                    Bx�4�B  �          @�
=�b�\?!G�@p  B8�HC)�f�b�\��@q�B:p�C;ff                                    Bx�4��  �          @�z��c33>��R@n{B8�
C/��c33�B�\@i��B4�RC@                                    Bx�4��  �          @�p��Vff>�@z=qBD��C,5��Vff�(��@xQ�BB��C?8R                                    Bx�54  �          @��[�>��H@uB?�
C+�)�[��(�@u�B>�C>)                                    Bx�5�  "          @��R�Z=q?��@x��BA
=C*{�Z=q��@y��BA��C<��                                    Bx�5)�  <          @�p��k�?��@c�
B.��C+�
�k���@dz�B/�\C;\)                                    Bx�58&  �          @����|(�>���@^{B$C.0��|(��z�@\(�B#=qC<n                                    Bx�5F�  �          @�Q����H>B�\@S33B��C1aH���H�=p�@N{BG�C>J=                                    Bx�5Ur  T          @�(��n�R>��@b�\B-�
C1���n�R�W
=@\(�B'�
C@��                                    Bx�5d  �          @��H�W�?��@q�B?ffC*���W���@q�B?�RC<��                                    Bx�5r�  �          @�G��Z=q?��@j�HB:C*���Z=q��@k�B;ffC<{                                    Bx�5�d  �          @�{�Z=q>�33@c33B8  C.+��Z=q�&ff@`��B5=qC>��                                    Bx�5�
  �          @�  �Vff>�ff@l(�B>
=C,aH�Vff�
=@j�HB<�RC=�                                    Bx�5��  T          @�G��g�>��
@uB:ffC/�g��@  @q�B6�\C?�                                     Bx�5�V  �          @��e�>\@n�RB8�C.  �e��(��@l(�B5��C>xR                                    Bx�5��  T          @�(��_\)>�z�@n�RB;G�C/L��_\)�@  @j�HB7�C@�                                    Bx�5ʢ  "          @��
�^�R>�Q�@o\)B;z�C.)�^�R�+�@l(�B8��C>�3                                    Bx�5�H  �          @���aG�>��@j�HB8  C-J=�aG����@i��B6Q�C=�                                    Bx�5��  �          @�(��j=q>�{@e�B1{C.� �j=q�&ff@b�\B.ffC>\                                    Bx�5��  �          @�33�e�?��@e�B2ffC+O\�e���G�@eB3Q�C;�                                    Bx�6:  �          @�Q��XQ�k�@j=qB<�RC7޸�XQ쿚�H@]p�B/CG��                                    Bx�6�  
�          @��\�W����@\(�B6�C8xR�W���
=@O\)B){CGQ�                                    Bx�6"�  �          @����,(���  @{�B]�
C9G��,(���ff@n{BL�
CM��                                    Bx�61,  
�          @����*=q�&ff@w
=BZ��CA�q�*=q��z�@b�\BB
=CS�q                                    Bx�6?�  �          @|(����>aG�@>{BIz�C.�\��Ϳz�@:�HBE  CAO\                                    Bx�6Nx  T          @�������Q�?���B?��C7��������?\B3  CG��                                    Bx�6]  "          @
�H��׾���?5A��HC<���׾�?(�A��
CB8R                                    Bx�6k�  
�          ?�>�����}p���HC��R>��z�H��ff�H�HC��)                                    Bx�6zj  "          ?�=q��ff�O\)>�G�A�  Cq���ff�fff>W
=A>ffCs��                                    Bx�6�  "          @�33�U���{@UB.��CFk��U�����@;�B
=CRff                                    Bx�6��  T          @���Y���.{@a�B8G�C6��Y������@W
=B-{CE��                                    Bx�6�\  T          @�=q�c33��\@\)A�Q�CQ���c33�'�?�{A�G�CXxR                                    Bx�6�  �          @�ff�S�
�+�@XQ�B4  C?ff�S�
���
@EB �CLǮ                                    Bx�6è  �          @����c�
��(�@c�
B3G�C:�f�c�
����@U�B$��CH��                                    Bx�6�N  T          @��H�`  ��{@j�HB8�HC9���`  ���@]p�B+=qCH&f                                    Bx�6��  "          @����mp��k�@Z�HB*��C7�{�mp���{@O\)B�HCD��                                    Bx�6�  �          @�z��o\)=���@`��B,��C2p��o\)�O\)@Z�HB&�C@8R                                    Bx�6�@  �          @�ff�r�\>u@b�\B+��C0n�r�\�0��@^�RB(ffC>E                                    Bx�7�  T          @�  ��G�>u@VffBp�C0�
��G��!G�@S33B�\C<޸                                    Bx�7�  �          @�
=��z�>W
=@J=qB(�C1#���z�(�@G
=B=qC<ff                                    Bx�7*2  T          @�p����=���@5BffC2����녿!G�@1G�BC<Q�                                    Bx�78�  �          @�{�xQ콣�
@[�B%�C5!H�xQ�s33@S33B(�CA�                                    Bx�7G~  T          @�  �h��=���@:=qBffC2z��h�ÿ&ff@5B��C>
                                    Bx�7V$  �          @�p��[���\)@C33B&��C5&f�[��W
=@;�B  CA��                                    Bx�7d�  �          @Y���#�
<�@p�B#��C3k��#�
�
=q@	��BC?ٚ                                    Bx�7sp  T          @G���\)��
=���R�&��CBn��\)����ff�.�
C4�                                    Bx�7�  �          @7����þ�G����l�CF8R����>.{���rz�C,�\                                    Bx�7��  �          @c�
>Ǯ�   �[�{C��R>Ǯ>\�\��¢B�B1Q�                                    Bx�7�b  �          @x��?��
�
=q�]p��3C�U�?��
>�33�^�Rp�AL                                      Bx�7�  T          @�z�?��Ϳ
=�j=q�x  C�q?���>�{�l(��{�
A&ff                                    Bx�7��  �          @u�?�Q���QG��i��C�aH?�Q�>����S33�m{A�H                                    Bx�7�T  �          @�G�@�
����e��b�C�G�@�
>�G��e�c  A,                                      Bx�7��  �          @���@"�\=#�
���j��?k�@"�\?�=q��G��^��A��
                                    Bx�7�  �          @��R?�{��G���=q��C�Q�?�{?333��G��A�p�                                    Bx�7�F  T          @���?�\)�G����HG�C���?�\)>�p���z�B�A4Q�                                    Bx�8�  T          @��@�׿(����H�|
=C�xR@��?
=q���H�|�
AW
=                                    Bx�8�  "          @�{@��=p������r�\C���@�>Ǯ����v��AG�                                    Bx�8#8  �          @��
?��R�@  �����C���?��R>����{G�A:�R                                    Bx�81�  �          @�{@G������\)�C���@G�?�����A|                                      Bx�8@�  �          @�@
=�����R�RC��3@
=?.{��k�A���                                    Bx�8O*  �          @�
=@33�\�����u�C�O\@33?+���  �r=qA��                                    Bx�8]�  T          @��H@ �׿&ff��=q�f  C��{@ ��>�33����iQ�@�ff                                    Bx�8lv  �          @��@��\)��33�j�C�w
@�>�G�����l{A$z�                                    Bx�8{  T          @��R@0�׿O\)�n{�QQ�C���@0��=�G��s�
�X\)@��                                    Bx�8��  "          @�(�@=p��^�R�\���@��C��3@=p��#�
�c�
�I�C���                                    Bx�8�h  �          @�z�@@  �O\)�Fff�3�C�ff@@  �L���L���;�\C��                                    Bx�8�  
(          @��@Dz�Y���J=q�2�C�@ @Dz὏\)�QG��;�C�N                                    Bx�8��  "          @���@.�R����G
=�A�C�"�@.�R>�\)�HQ��B��@�                                    Bx�8�Z  T          @��
@C�
����0  �&p�C��)@C�
>B�\�1��(��@fff                                    Bx�8�   �          @��@0��>��
�AG��=�@ָR@0��?��
�7
=�0�A�                                    Bx�8�  	�          @g�@(Q�fff����33C���@(Q쾸Q��ff�%�HC�                                      Bx�8�L  �          @mp�@333�^�R�  ��C�W
@333���
����!=qC��)                                    Bx�8��  �          @/\)?�(���׿��'�C�O\?�(�����\)�-�C���                                    Bx�9�  T          @{?�(��#�
��\)�p�C��3?�(�>��R����A#�                                    Bx�9>  �          @G�?�Q�=��Ϳ�z���  @AG�?�Q�>�p���{����A-p�                                    Bx�9*�  �          @�?�p�������(�C�N?�p�>W
=�����
�@���                                    Bx�99�  �          @6ff@  �L�Ϳ�(��=qC���@  >W
=��(��{@���                                    Bx�9H0  �          @B�\@  ��Ϳ�(��!�C�)@  ���
��\�(��C��{                                    Bx�9V�  �          @L(�@ff�n{��
=��C�+�@ff����$��C�l�                                    Bx�9e|  �          @K�@
=��z��\)�p�C�R@
=�p������+Q�C��=                                    Bx�9t"  T          @Fff?�{������Q�C�3?�{��{���%z�C���                                    Bx�9��  �          @:=q?�p���\������\C�+�?�p����׿�33�"C��f                                    Bx�9�n  �          @#33?��R��p��������C��
?��R��\)�޸R�,��C�
                                    Bx�9�  �          ?���?5��\)�h����33C���?5���׿�Q��'=qC���                                    Bx�9��  �          ?�?L�Ϳ��׿h����C���?L�Ϳc�
��33�.ffC��q                                    Bx�9�`  �          ?����\�   >\)A2�HC`��\��<�@ ��Ca&f                                    Bx�9�  �          ?�p��h�ÿ�  >��A��
Cc�Ϳh�ÿ��>8Q�@�  Ceh�                                    Bx�9ڬ  �          @$z��ff���
?�z�B*�\C6����ff��ff?���B"�CB!H                                    Bx�9�R  �          @|���-p�?J=q@.�RB0{C#���-p�>#�
@5�B8�C0��                                    Bx�9��  �          @|(��.{?#�
@*�HB.�RC&�\�.{<�@0  B4��C3h�                                    Bx�:�  
�          @h���333<#�
@�\BQ�C3Ǯ�333�   @\)B�HC>�                                    Bx�:D  T          @W�� ��>�@B{C0�� �׾���@z�B{C;�                                    Bx�:#�  �          @���1G�?�  @HQ�B7p�C��1G�?�@U�BG  C)(�                                    Bx�:2�  �          @����8Q�?�{@W
=B:�C��8Q�?
=@e�BJ\)C(h�                                    Bx�:A6  T          @�p��B�\?��@S33B3�RC��B�\?
=q@`  BB=qC)�q                                    Bx�:O�  �          @����N{?��\@L(�B-�C"s3�N{>���@UB7C.�                                     Bx�:^�  
�          @�z��L��?u@Mp�B/�\C#=q�L��>u@UB8�
C/�3                                    Bx�:m(  �          @��N�R?��@Mp�B-p�C!���N�R>�{@W
=B8{C.                                      Bx�:{�  �          @�  �Tz�?���@K�B'�
C ��Tz�>��H@W
=B4(�C+��                                    Bx�:�t  �          @����Q�?��@G�B(�C"#��Q�>�Q�@QG�B3(�C-                                    Bx�:�  �          @��R�Z=q?�  @EB$
=C#�
�Z=q>��
@O\)B-z�C.��                                    Bx�:��  T          @�\)�]p�?u@Dz�B"G�C$s3�]p�>�\)@Mp�B+
=C/Y�                                    Bx�:�f  �          @�{�fff?c�
@7
=Bp�C&��fff>��@>�RBG�C/�
                                    Bx�:�  T          @�{�r�\?@  @(��B	ffC(�{�r�\>.{@/\)BQ�C1s3                                    Bx�:Ӳ  �          @�(��o\)?:�H@'�B	C(��o\)>#�
@-p�B��C1�=                                    Bx�:�X  
�          @�ff�p��?=p�@-p�B
=C(�f�p��>��@3�
BC1�                                     Bx�:��  �          @��R�mp�?L��@1�B�C'�\�mp�>L��@8Q�B33C0�3                                    Bx�:��  "          @���a�?(��@3�
B=qC)p��a�=L��@8Q�B{C3&f                                    Bx�;J  �          @���b�\?8Q�@7
=Bp�C(p��b�\=�G�@<��B�C2=q                                    Bx�;�  T          @���j�H?+�@333Bz�C)���j�H=�\)@8Q�B\)C2�
                                    Bx�;+�  �          @���hQ�?E�@3�
B�RC(��hQ�>.{@:=qB�C1\)                                    Bx�;:<  T          @�
=�\��?�
=@@  BC!
�\��?�@K�B)Q�C+5�                                    Bx�;H�  T          @���s33?:�H@%�B
=C)
�s33>8Q�@+�BC1E                                    Bx�;W�  �          @�(��w
=?�@\)BG�C,\)�w
=�#�
@"�\Bp�C4�                                    Bx�;f.  T          @��
�p  ?c�
@"�\BffC&���p  >�33@*�HB(�C.�f                                    Bx�;t�  �          @��\�k�?W
=@%B
  C'#��k�>�z�@-p�B33C/}q                                    Bx�;�z  �          @���i��?c�
@+�B
=C&5��i��>���@333B�
C.ٚ                                    Bx�;�   T          @�z��hQ�?B�\@2�\B��C(33�hQ�>.{@8��B��C1B�                                    Bx�;��  �          @����i��?@  @1�B�C(ff�i��>.{@8Q�B��C1W
                                    Bx�;�l  �          @�Q��q�?Y��@0��B�HC'ff�q�>�=q@8Q�B�RC/�                                    Bx�;�  �          @����s�
?���@*=qB�RC$@ �s�
?�\@4z�B33C,J=                                    Bx�;̸  
�          @�z��n�R?�33@ ��B�C"���n�R?�R@,(�B��C*��                                    Bx�;�^  
�          @�G��tz�?�{@*=qB33C#�)�tz�?��@5�B  C+�\                                    Bx�;�  "          @�
=�p  ?z�H@,��BffC%n�p  >�
=@5B��C-��                                    Bx�;��  �          @�z��|��?�z�@Q�A�G�C#���|��?8Q�@z�A��\C)�R                                    Bx�<P  "          @�z��w�?�(�@  A�z�C"���w�?=p�@��A���C)�                                    Bx�<�  �          @��x��?��@\)A���C!\)�x��?W
=@��A��C'                                    Bx�<$�  �          @�����Q�?�ff@�RA�\)C"  ��Q�?W
=@��A���C(5�                                    Bx�<3B  �          @�  �n�R?�
=@#�
B��C�q�n�R?fff@333B�C&ff                                    Bx�<A�  �          @����|��?��H@'�BG�C"��|��?.{@333B��C*E                                    Bx�<P�  �          @����z�?�ff@=qA뙚C%����z�?\)@$z�A��RC,W
                                    Bx�<_4  �          @�������?�G�@
=A�G�C%�3����?�@ ��A�(�C,�                                    Bx�<m�  �          @��\)?���@
�HAݙ�C$�=�\)?(��@ffA��HC*�
                                    Bx�<|�  �          @�
=����?��\@p�A�(�C%Ǯ����?z�@�A�C+��                                    Bx�<�&  �          @����u?�p�@"�\B ��C"@ �u?8Q�@.�RB�HC)k�                                    Bx�<��  �          @��\�w�?�G�@&ffBG�C!��w�?=p�@2�\BffC)+�                                    Bx�<�r  �          @��\�tz�?��
@+�B�C!s3�tz�?=p�@8Q�B��C(��                                    Bx�<�  �          @�z��p��?�\)@)��B�RC�3�p��?��@:=qB��C#��                                    Bx�<ž  "          @��z=q?�Q�@'�B ffC���z=q?k�@5B{C&�q                                    Bx�<�d  �          @�{���\?���@��A��C!O\���\?fff@'�B Q�C'��                                    Bx�<�
  "          @���=q?�G�@�RA�C"ٚ��=q?E�@*�HB\)C)Y�                                    Bx�<�  "          @�\)����?��@p�A��HC0�����?�ff@-p�B�C%z�                                    Bx�= V  "          @�  ���\?�p�@p�A�(�C )���\?}p�@,(�B��C&\)                                    Bx�=�  "          @�G��{�?�\)@<(�B{C$(��{�?��@FffB  C,
                                    Bx�=�  �          @��\��  ?�  @7�B	�RC"����  ?0��@C33B�HC*(�                                    Bx�=,H  �          @����~{?���@333B=qC!�=�~{?G�@@  B33C(ٚ                                    Bx�=:�  �          @�Q��~{?�G�@1�B�C"k��~{?8Q�@>{B33C)��                                    Bx�=I�  �          @������\?��H@*=qB �C#k����\?333@5B	�
C*5�                                    Bx�=X:  �          @������?��\@ ��AC#  ���?J=q@,��B  C)=q                                    Bx�=f�  "          @�Q���(�?�\)@\)A�ffC!�)��(�?c�
@,��B\)C'Ǯ                                    Bx�=u�  
�          @�����{?���@{A�=qC"0���{?^�R@*�HA�C(8R                                    Bx�=�,  �          @��\��  ?���@(�A�  C"Ǯ��  ?Y��@(��A��RC(��                                    Bx�=��  �          @��
���?���@&ffA���C �R���?u@4z�B
=C'�                                    Bx�=�x  T          @��\����?�G�@,(�B =qC}q����?�  @:�HB��C&)                                    Bx�=�  �          @�z���33?��R@,��A�z�C ���33?z�H@;�B33C&��                                    Bx�=��  �          @�33����?�Q�@1�B(�C Q�����?k�@?\)BC'&f                                    Bx�=�j  �          @��
����?�Q�@1�B�C }q����?k�@?\)B��C'@                                     Bx�=�  �          @�p���p�?��@%�A�C}q��p�?�z�@5�B�
C$xR                                    Bx�=�  �          @�\)�z=q?�(�@@  BG�C:��z=q?�z�@P��BffC#p�                                    Bx�=�\  T          @�{��=q?���@333B��C���=q?�ff@A�B�C%��                                    Bx�>  �          @��R��
=?��
@(��A�C )��
=?��@7�B�C&.                                    Bx�>�  T          @�
=��z�?Ǯ@/\)A�=qC\)��z�?�ff@>{BG�C%��                                    Bx�>%N  �          @�ff�qG�?��@Tz�B�
C s3�qG�?=p�@`  B)�C(�                                    Bx�>3�  �          @���}p�?��@8Q�Bp�Cs3�}p�?�\)@G�B�C$:�                                    Bx�>B�  
�          @�z���G�?�@,��A�CxR��G�?�
=@<��B33C#�3                                    Bx�>Q@  �          @�ff�vff?\@I��B
=Cff�vff?u@W
=B!=qC&�                                    Bx�>_�  �          @�p��fff?xQ�@e�B/\)C$��fff>�{@l(�B6p�C.��                                    Bx�>n�  "          @��R�b�\?xQ�@l(�B4�\C$�f�b�\>��
@s33B;�C.�                                    Bx�>}2  �          @�\)�c�
?p��@n{B5�C%(��c�
>�z�@u�B;�
C/J=                                    Bx�>��  T          @�  �|(�?�\)@L(�B�\C ���|(�?O\)@XQ�B 
=C(ff                                    Bx�>�~  �          @������?�ff@J=qB��C�����?�  @XQ�BffC&!H                                    Bx�>�$  "          @�����H?�ff@=p�B
=CB����H?��\@N{BC"�R                                    Bx�>��  �          @��H���@ ��@/\)A���C����?\@B�\B
��C��                                    Bx�>�p  "          @�G�����?��@6ffB��C�����?�\)@HQ�B{C!0�                                    Bx�>�  "          @����~{?�z�@Dz�B  CO\�~{?�\)@S33BC$5�                                    Bx�>�  �          @�������?�  @:�HBQ�C������?��R@J�HB�C"�                                    Bx�>�b  �          @�z����?�@@��B(�C�
���?���@Q�B
=C"
=                                    Bx�?  �          @�z���=q?�{@@  B�C���=q?�=q@P��B
=C!�H                                    Bx�?�  �          @�(����H?�z�@;�B�HC�q���H?�33@L��B(�C!�                                    Bx�?T  �          @�(���  ?�(�@@  B��CǮ��  ?���@Q�B�C �                                    Bx�?,�  �          @�����=q?��@N{B�\C33��=q?�=q@^�RBffC!�                                    Bx�?;�  �          @�����G�?��R@N�RB�CǮ��G�?�Q�@aG�B\)C p�                                    Bx�?JF  �          @�  ����?�33@N{Bp�C�q����?���@_\)Bz�C!s3                                    Bx�?X�  �          @�������@�\@@��B=qC�=����?��
@S33B  C�=                                    Bx�?g�  �          @����(�@ff@Dz�B��C���(�?�=q@W
=B{C�                                    Bx�?v8  �          @����\)@p�@B�\B�C\)��\)?�Q�@VffBz�C#�                                    Bx�?��  �          @����@Q�@J�HB(�C�R��?���@^{B33C
                                    Bx�?��  
Z          @�p����\@�
@VffB��C:����\?�  @hQ�B�Cٚ                                    Bx�?�*  �          @�ff����@
=@\(�BQ�CO\����?��@n�RB#z�C
                                    Bx�?��  �          @�p���  @
�H@X��B��Cu���  ?�{@l(�B"G�C�                                    Bx�?�v  �          @�(���  @�
@Mp�B33C�q��  ?�\@b�\B�RC{                                    Bx�?�  "          @�(�����@��@K�B	�RC�\����?�p�@`  BC��                                    Bx�?��  T          @����z�@z�@Dz�B��C�R��z�?�@Y��BCaH                                    Bx�?�h  "          @��
��33@Q�@A�BQ�C�)��33?��@W
=B�Cc�                                    Bx�?�  T          @��H��33@
=@>{B ��C\��33?�\)@S33B
=C}q                                    Bx�@�  �          @�33��z�@��@:�HA��C���z�?�z�@P  B  CB�                                    Bx�@Z  �          @����ff@�@4z�A�\C���ff?���@H��B	�HC�H                                    Bx�@&   �          @������@=q@0��A��C����?���@FffB
=Cٚ                                    Bx�@4�  �          @����(�@��@+�A��C^���(�@ ��@AG�BQ�C�                                    Bx�@CL  T          @��
�~{@5@��A��
Cff�~{@{@'
=A�
=C{                                    Bx�@Q�  "          @�z���(�@:�H?�=qA9p�C=q��(�@.�R?�G�A���C{                                    Bx�@`�  T          @������@AG�?aG�A�C�=����@6ff?�=qAc�C\                                    Bx�@o>  �          @��H��p�@.{?���APQ�C^���p�@ ��?���A���Ch�                                    Bx�@}�  "          @����
=@-p�?�{Ai�C����
=@�R?�G�A��
C                                      Bx�@��  �          @����=q@@  ?�G�AX  C:���=q@1G�?ٙ�A�z�CG�                                    Bx�@�0  �          @�ff��
=@*=q?���A���CB���
=@��?�p�A���C�H                                    Bx�@��  �          @�{��{@,(�?��A�  C��{@�H@G�A�=qCk�                                    Bx�@�|  T          @�����@'
=?�33A��
C� ��@@G�A�33C0�                                    Bx�@�"  �          @�p����R@�?�\)A��CxR���R@�@p�A�z�C�\                                    Bx�@��  �          @�p���@z�@z�A�  Ch���?��R@��AѮC޸                                    Bx�@�n  �          @��
��@�R@�A��
CO\��?�33@�A�ffC��                                    Bx�@�  �          @�z�����@�H?�Q�A��HC33����@
=@G�AǮCc�                                    Bx�A�  T          @�z����@\)@z�A��\C�R���@
=q@=qA�z�CY�                                    Bx�A`  
�          @����z�@�R@�A�  C
��z�@
=@*=qA�z�C�3                                    Bx�A  �          @����
@(Q�@
=A�ffCs3���
@��@.{A�ffC@                                     Bx�A-�  
�          @�{���@   @��A�z�C  ���@�@2�\A���C                                      Bx�A<R  "          @���ff@��@p�A׮CJ=��ff@G�@1�A�ffCW
                                    Bx�AJ�  �          @�p�����@@Q�A�{CT{����?�(�@,(�A�C:�                                    Bx�AY�  �          @�����@�@��A�{Cff��@   @0��A��\Ck�                                    Bx�AhD  �          @��
���
@Q�@\)A�p�C�q���
@   @3�
A�{C�                                    Bx�Av�  "          @�z����H@@'
=A�C:����H?���@:=qB��C��                                    Bx�A��  T          @�(���z�@�@!G�A�33C�f��z�?���@4z�A�
=CǮ                                    Bx�A�6  �          @��
����@��@ffA��C
����?�z�@(��A�C��                                    Bx�A��  �          @����  @�\@�AΏ\C����  ?�Q�@(Q�A�G�Cs3                                    Bx�A��  T          @�33����@��@
=AѮC�R����?���@)��A�33C��                                    Bx�A�(  
Z          @����ff@��@\)A݅Ck���ff?�=q@1G�A��C}q                                    Bx�A��  �          @��\���@@z�A���C&f���?�  @%A��HC�                                    Bx�A�t  �          @�����
=@��@%�A�Q�C)��
=?�\@6ffA���CO\                                    Bx�A�  �          @�{����@33@(Q�A��Cu�����?�@8��B �C�q                                    Bx�A��  �          @�
=����@z�@*�HA��
CQ�����?�
=@;�B�C��                                    Bx�B	f  �          @�
=����@33@.{A�RCs3����?�z�@>�RB��Cٚ                                    Bx�B  �          @�{��p�@
=@1�A�(�C8R��p�?��H@B�\B{C�q                                    Bx�B&�  �          @��R��{@G�@7�A���C=q��{?�\)@G
=B
�C�                                    Bx�B5X  �          @�������@ ��@C33B��C0�����?�=q@S33BG�C0�                                    Bx�BC�  �          @�G���33?���@J�HB��C�H��33?�  @Z=qBC��                                    Bx�BR�  �          @�G�����@   @E�BQ�CB�����?���@Tz�Bz�CL�                                    Bx�BaJ  �          @�����{@G�@A�Bp�CE��{?���@Q�B��C&f                                    Bx�Bo�  "          @�����G�@�\@5�A�=qC����G�?��@Dz�B33C�                                    Bx�B~�  �          @�Q�����?��R@*=qA�\C������?�\)@9��A��Cٚ                                    Bx�B�<  "          @�=q����@ ��@E�B�C&f����?˅@S�
BC�                                    Bx�B��  S          @��\��(�@ff@FffB33C���(�?�
=@VffB��C�                                    Bx�B��  �          @����ff@	��@=p�B \)C�f��ff?޸R@Mp�B
=Cu�                                    Bx�B�.  �          @�=q����@�
@E�BffC�\����?��@Tz�B��Cc�                                    Bx�B��  �          @��H��
=@ ��@C33B�\Cu���
=?���@R�\B\)C5�                                    Bx�B�z  �          @��\���?�Q�@J=qB
Q�C����?\@X��B��C��                                    Bx�B�   �          @�����=q?��R@N{B�\C�f��=q?Ǯ@\��Bp�C                                    Bx�B��  �          @�=q����?�
=@R�\B��Cu�����?��R@`��Bz�C�q                                    Bx�Cl  T          @�=q��G�?�@X��B��C�H��G�?�{@eB �\C!h�                                    Bx�C  �          @��H���?�ff@X��B��C����?���@eB 33C!��                                    Bx�C�  "          @����Q�?�@X��B�\C����Q�?�{@fffB!Q�C!8R                                    Bx�C.^  �          @��H�z=q?�  @eB�
C�)�z=q?��
@r�\B*ffC!ٚ                                    Bx�C=  �          @��\�xQ�?޸R@g�B!CǮ�xQ�?��\@s�
B,Q�C!ٚ                                    Bx�CK�  �          @�G��w
=?�ff@c33B�C
=�w
=?��@p  B*  C �                                    Bx�CZP  �          @���x��?޸R@dz�B��C���x��?��
@p��B*{C!��                                    Bx�Ch�  �          @�=q�~�R?��@a�Bz�C���~�R?�
=@mp�B'{C#s3                                    Bx�Cw�  "          @�Q�����?�z�@N{BQ�C������?��R@[�Bp�C�3                                    Bx�C�B  �          @�Q�����?�p�@J=qB
=C������?���@XQ�Bz�C��                                    Bx�C��  
�          @�=q����@33@P��B�C�����?У�@_\)BQ�C��                                    Bx�C��  T          @����s�
@�\@\��BffC�H�s�
?˅@j�HB&��CG�                                    Bx�C�4  
�          @���Vff?���@��
BC
=C  �Vff?W
=@�Q�BK��C%��                                    Bx�C��  "          @����Tz�?���@�{BE��Cz��Tz�?Y��@��\BN�\C%��                                    Bx�Cπ  �          @�Q��[�?�
=@�=qB>�CL��[�?n{@�
=BG��C$��                                    Bx�C�&  "          @���i��?��@h��B&{C���i��?�@uB1�\C�q                                    Bx�C��  T          @���j=q?��
@k�B(\)C��j=q?���@w�B333C !H                                    Bx�C�r  "          @�Q��l(�?�(�@mp�B)Q�C�l(�?�G�@x��B3�RC!(�                                    Bx�D
  �          @��R�l��?��@fffB$��C#��l��?���@r�\B/��C                                       Bx�D�  �          @�
=�j�H?�=q@g�B%��C�{�j�H?���@s�
B0��Cn                                    Bx�D'd  "          @�\)�g
=?�Q�@p  B,�C��g
=?�p�@{�B733C!33                                    Bx�D6
  "          @�
=�^�R?�(�@|��B9(�C)�^�R?}p�@�33BBG�C$&f                                    Bx�DD�  "          @��R�X��?�Q�@���B>(�C���X��?s33@�p�BG=qC$O\                                    Bx�DSV  
�          @��R�Q�?�(�@��BC\)C�{�Q�?xQ�@�Q�BL�
C#p�                                    Bx�Da�  T          @��R�HQ�?��R@�G�BOffCT{�HQ�?:�H@��BW�C&�H                                    Bx�Dp�  �          @���>{?�Q�@�ffBY  C(��>{?(��@��Ba  C'z�                                    Bx�DH  �          @����2�\?�p�@�33Ba�C33�2�\?.{@��RBj��C&@                                     Bx�D��  "          @�Q��0  ?��\@���Bf��C�)�0  >��@��Bm��C*8R                                    Bx�D��  
�          @�Q��6ff?��
@��Bb��C .�6ff>�@�{Bi��C*\)                                    Bx�D�:  �          @�Q��3�
?�ff@�(�Bd�C���3�
?   @�
=Bk33C)�{                                    Bx�D��  T          @�  �7
=?�{@��B`{C�{�7
=?z�@��Bg�C(��                                    Bx�DȆ  �          @�  �>{?�Q�@�
=BY��C+��>{?+�@��\Ba�C'W
                                    Bx�D�,  
�          @�  �Dz�?��@��
BR�
C33�Dz�?G�@��B[=qC%�q                                    Bx�D��  �          @�\)�2�\?k�@�(�BfffC!� �2�\>\@�ffBl\)C,&f                                    Bx�D�x  �          @�\)�,(�?p��@�Bj��C ���,(�>���@�  Bp��C+z�                                    Bx�E  
�          @�{�Q�?G�@�=qBz{C!��Q�>k�@�(�BffC.��                                    Bx�E�  �          @�p���?:�H@�=qB{
=C"�=��>8Q�@��
B��C/�\                                    Bx�E j  	.          @�p��33?.{@�33B~�
C#u��33>�@���B���C0�q                                    Bx�E/  �          @����Q�?+�@���Bz�HC$5��Q�>�@��HB(�C0��                                    Bx�E=�  �          @���=q?�R@��Bz�C%���=q=�\)@�33B~Q�C2E                                    Bx�EL\  T          @�p��=q?#�
@��Bz(�C%0��=q=���@�33B~  C1�                                    Bx�E[  �          @����=q?
=@���Bz��C&+��=q=u@��HB~  C2�                                    Bx�Ei�  �          @�(��z�?��@��\B~��C&���z�<#�
@��B��fC3�=                                    Bx�ExN  
�          @�����?   @��\B��qC'����׽#�
@��B�
=C4��                                    Bx�E��  �          @���?   @���B~�C(  ���@�=qB�G�C4�)                                    Bx�E��  �          @��
��>�@��B~��C(xR���u@��\B�p�C5Y�                                    Bx�E�@  
�          @��
��?�\@�(�B�ǮC&������@��B��C4��                                    Bx�E��  
�          @��H�{>�@��\B���C(Q��{��\)@�33B��qC5�\                                    Bx�E��  T          @�33��>��@�=qB�z�C(c��녽u@��HB��{C5}q                                    Bx�E�2  
�          @�33�z�>\@��B���C)�f�z�#�
@�p�B�G�C8W
                                    Bx�E��  T          @��H����>�33@��B��qC)!H���;B�\@�  B�G�C9��                                    Bx�E�~  �          @�=q��ff>�p�@��B��C(J=��ff�.{@�  B�8RC9G�                                    Bx�E�$  �          @��
�Q�>��R@�p�B�ǮC+���Q�aG�@�p�B�
=C9�H                                    Bx�F
�  �          @���G�>���@�ffB�(�C+���G��k�@��RB�aHC:��                                    Bx�Fp  �          @��H�   >��@�ffB��)C/�q�   �\@�{B�(�C>                                    Bx�F(  �          @�ff�>��@�  B�ffC/�����Q�@��B�ǮC=                                    Bx�F6�  �          @�z��(�>��R@�(�B�\C,��(��B�\@�z�B�aHC8�H                                    Bx�FEb  �          @�z��?.{@���Bv�RC#���>L��@�=qB{=qC/\                                    Bx�FT  
�          @���p�?n{@�=qBj�CT{�p�>��@�z�Bq=qC)=q                                    Bx�Fb�  T          @����?@  @�G�Bm\)C"����>�z�@��HBrffC-(�                                    Bx�FqT  �          @������?#�
@��
Br  C%����>8Q�@��Bv
=C/Ǯ                                    Bx�F�  �          @����"�\?J=q@�  Bg�RC"�H�"�\>�33@��Bm  C,=q                                    Bx�F��  T          @����7
=?��@}p�BP\)CE�7
=?8Q�@���BW�RC%�
                                    Bx�F�F  
�          @���8��?�=q@�  BQQ�CxR�8��?(��@��HBX33C'�                                    Bx�F��  "          @����Dz�?�Q�@n�RB>�C�H�Dz�?�ff@w
=BG��C!.                                    Bx�F��  �          @����8��?��@y��BKQ�C���8��?fff@�Q�BSC"�                                    Bx�F�8  
�          @�G��   ?c�
@�  BaC s3�   >��@�=qBg��C)T{                                    Bx�F��  �          @�  �  ?
=q@���Bs\)C&c��  =�@�BvC1\                                    Bx�F�  
�          @�ff�Ǯ�#�
@�B���C9�
�Ǯ�(�@�z�B�.CIn                                    Bx�F�*  	�          @��Ϳ�=q��@��
B�W
C5���=q��@��HB�z�CD��                                    Bx�G�  �          @���{>aG�@�G�B��HC-���{�W
=@�G�B��C:��                                    Bx�Gv  �          @��R�  ?0��@�=qBo�
C"���  >�=q@��
Bt�C-)                                    Bx�G!  
�          @�Q��ff?+�@��\BlG�C$#��ff>�  @�(�Bp�RC-�f                                    Bx�G/�  T          @�����>�@���BnG�C)33���=#�
@�p�Bp��C3(�                                    Bx�G>h  T          @��
� ��>��H@�z�Bi{C(�H� ��=�\)@�p�Bk�C2O\                                    Bx�GM  
�          @��%?��@��Bf{C'��%>�@�{Bi�C1\                                    Bx�G[�  
�          @���'�?(�@��Bc(�C&ٚ�'�>L��@���Bf��C/��                                    Bx�GjZ  T          @��
�$z�?#�
@��\Bd33C%���$z�>k�@�(�Bh
=C.�H                                    Bx�Gy   "          @�33�#33?333@�=qBd  C$���#33>�z�@��
Bh\)C-k�                                    Bx�G��  
�          @����(�?&ff@��\Bh�
C%.�(�>u@��
Bl�HC.n                                    Bx�G�L  
�          @�����?&ff@�Bq�C$(���>k�@�
=Bu��C.8R                                    Bx�G��  T          @����33?5@��
BnC"� �33>���@�p�Bs��C,z�                                    Bx�G��  	�          @�Q���
?@  @��HBm33C"���
>�{@���Br\)C+��                                    Bx�G�>  �          @�Q���?5@��
Bo��C"����>���@�p�Bt�C,n                                    Bx�G��  �          @�  ��?0��@���Bg(�C$B���>�z�@��Bk��C-:�                                    Bx�Gߊ  
�          @�
=�%�?O\)@w�B\G�C"�H�%�>�
=@{�Baz�C*��                                    Bx�G�0  "          @�ff�;�?c�
@c33BD�
C#��;�?
=q@g�BJ�C)�\                                    Bx�G��  �          @�ff�3�
?B�\@l(�BN�HC$�
�3�
>���@o\)BSffC+�                                    Bx�H|  �          @��,(�?5@p��BU��C%33�,(�>�{@tz�BZ=qC,�                                     Bx�H"  T          @���8��?z�H@hQ�BG�HC!=q�8��?�R@mp�BM�
C'��                                    Bx�H(�  T          @�Q��@��?s33@c�
BB
=C"s3�@��?��@h��BG��C(�                                    Bx�H7n  �          @�\)�:�H?p��@fffBF
=C"0��:�H?
=@j�HBK��C(�H                                    Bx�HF  �          @����:=q?W
=@k�BJQ�C#�
�:=q>�@p  BO33C*�=                                    Bx�HT�  	�          @�=q�A�?:�H@k�BG
=C&p��A�>�p�@n�RBJ��C,��                                    Bx�Hc`  �          @���<(�?W
=@mp�BJ(�C$  �<(�>�@q�BO  C*��                                    Bx�Hr  �          @���:�H?p��@l��BIQ�C"33�:�H?z�@qG�BN�
C(                                    Bx�H��  
�          @���;�?xQ�@l(�BHp�C!�f�;�?�R@qG�BN�C(!H                                    Bx�H�R  �          @�=q�<(�?��\@k�BF�C �{�<(�?+�@p��BL�C'33                                    Bx�H��  
Z          @�=q�6ff?��\@o\)BL�C ^��6ff?(��@tz�BRG�C'                                    Bx�H��  
�          @�=q�8Q�?p��@p  BLz�C!���8Q�?z�@u�BR  C(��                                    Bx�H�D  �          @��\�8Q�?k�@p��BL�C"8R�8Q�?\)@u�BR{C(��                                    Bx�H��  T          @��\�?\)?J=q@mp�BH��C%0��?\)>�G�@qG�BM(�C+�                                    Bx�Hؐ  	�          @���333?G�@uBS�\C$u��333>��@x��BX{C+�
                                    Bx�H�6  �          @�G��.�R?O\)@vffBU�HC#aH�.�R>�ff@y��BZC*��                                    Bx�H��  
(          @����%�?fff@z�HB\�C �\�%�?�@~�RBb=qC(u�                                    Bx�I�  
(          @����.{?Q�@w
=BV�C#J=�.{>�ff@z�HB[�C*�\                                    Bx�I(  
�          @����6ff?k�@p  BMz�C"��6ff?�@tz�BR�C(��                                    Bx�I!�  "          @���-p�?=p�@z=qBY=qC$�R�-p�>�p�@}p�B]�\C,(�                                    Bx�I0t  �          @��H�*�H?O\)@|��B[=qC#��*�H>�G�@�Q�B`�C*��                                    Bx�I?  	`          @�33�,��?B�\@|��BZffC$8R�,��>Ǯ@�  B^�
C+��                                    Bx�IM�  �          @��\�5?W
=@s�
BP��C#�=�5>�@w�BUp�C*\)                                    Bx�I\f  �          @�33�A�?aG�@k�BEp�C#�q�A�?
=q@o\)BJG�C)�f                                    Bx�Ik  �          @���4z�?aG�@vffBR33C"���4z�?�@z�HBWG�C)�{                                    Bx�Iy�  �          @�33�2�\?c�
@w
=BSQ�C"B��2�\?�@z�HBX�\C)0�                                    Bx�I�X  �          @��H�1G�?E�@y��BVp�C$s3�1G�>��@|��BZ�
C+��                                    Bx�I��  �          @�(��8Q�?k�@u�BN�C"O\�8Q�?\)@x��BT(�C(�                                    Bx�I��  �          @���<��?u@s33BKQ�C!޸�<��?(�@xQ�BPC(E                                    Bx�I�J  �          @�p��>{?���@n�RBEp�C�R�>{?Y��@tz�BLQ�C#�                                    Bx�I��  �          @����>�R?��R@l(�BCG�Ch��>�R?fff@r�\BJ\)C#:�                                    Bx�Iі  �          @�(��=p�?�Q�@l(�BD��C)�=p�?Y��@q�BK��C$�                                    Bx�I�<  �          @�33�5�?��R@o\)BI�HCc��5�?c�
@uBQ=qC"�                                    Bx�I��  �          @��\�0��?���@q�BM��Cn�0��?\(�@w�BU�C"�                                    Bx�I��  �          @���4z�?��
@j�HBG�\C���4z�?p��@q�BO33C!xR                                    Bx�J.  
�          @��\�;�?��@g
=BAz�C.�;�?u@mp�BH��C!�{                                    Bx�J�  �          @��\�8Q�?�
=@l(�BGz�C�R�8Q�?W
=@r�\BNQ�C#�R                                    Bx�J)z  �          @�=q�2�\?��
@r�\BO�C���2�\?0��@w�BU��C&(�                                    Bx�J8   �          @����4z�?��
@p  BM\)C���4z�?0��@u�BSffC&G�                                    Bx�JF�  �          @����7
=?aG�@n{BL��C"���7
=?
=q@q�BQ��C)E                                    Bx�JUl  �          @�  �8��?E�@k�BKz�C%��8��>�G�@n�RBO�C+p�                                    Bx�Jd  �          @���7�?:�H@l(�BM{C%���7�>Ǯ@o\)BQ  C,@                                     Bx�Jr�  �          @���8��?5@k�BK�C&!H�8��>\@n�RBO�RC,}q                                    Bx�J�^  �          @�  �<(�?.{@j=qBI��C&�f�<(�>�33@mp�BM�C-&f                                    Bx�J�  �          @�
=�3�
?@  @mp�BO��C$���3�
>�
=@qG�BS�
C+��                                    Bx�J��  �          @��R�3�
?E�@l(�BN�
C$�)�3�
>�G�@p  BS(�C+�                                    Bx�J�P  �          @�
=�5�?O\)@k�BM��C#���5�>�@o\)BR(�C*aH                                    Bx�J��  T          @�{�2�\?\(�@j�HBM��C"��2�\?�@n�RBR�HC)T{                                    Bx�Jʜ  �          @���-p�?xQ�@j=qBO33C E�-p�?#�
@n�RBU
=C&�                                    Bx�J�B  �          @�{�)��?s33@p  BTp�C (��)��?�R@tz�BZ=qC&޸                                    Bx�J��  �          @�  �(Q�?Q�@w�BZ\)C"�R�(Q�>��@{�B_33C)�)                                    Bx�J��  �          @�Q��)��?h��@vffBW�C!��)��?�@z=qB]  C'�                                    Bx�K4  �          @����+�?n{@vffBVG�C ���+�?z�@z�HB[��C'Ǯ                                    Bx�K�  �          @����,��?p��@w
=BU�RC ޸�,��?
=@z�HB[G�C'�f                                    Bx�K"�  �          @�Q��!G�?�ff@y��B[�HCG��!G�?5@~{Bb��C$Y�                                    Bx�K1&  �          @�\)�(��?n{@s33BV�C ���(��?
=@w�B\(�C'aH                                    Bx�K?�  �          @��R�,��?Y��@p��BS��C"u��,��?�@tz�BX��C)#�                                    Bx�KNr  �          @�
=�/\)?W
=@o\)BR
=C"�H�/\)?�\@s33BV�HC)s3                                    Bx�K]  �          @��(Q�?p��@p  BU�C \)�(Q�?��@tz�BZ�
C'�                                    Bx�Kk�  �          @��R�&ff?�ff@q�BV
=C��&ff?5@w
=B\�C$��                                    Bx�Kzd  �          @�
=�)��?aG�@s33BV��C!���)��?
=q@w
=B[�
C(u�                                    Bx�K�
  �          @�  �,(�?Y��@u�BVp�C"s3�,(�?�\@x��B[ffC)E                                    Bx�K��  �          @�
=�*=q?W
=@s�
BWG�C"��*=q?   @w�B\33C)aH                                    Bx�K�V  �          @�{�#�
?5@w
=B]��C$�=�#�
>�p�@z=qBb�C+�H                                    Bx�K��  �          @�ff�#�
?=p�@w�B]��C#ٚ�#�
>���@z�HBb\)C+.                                    Bx�Kâ  �          @�
=�,(�?O\)@r�\BU�RC#Y��,(�>��@uBZ\)C*!H                                    Bx�K�H  �          @��R�%�?B�\@w�B\�C#���%�>�
=@z�HBa  C*��                                    Bx�K��  �          @�{�
=?^�R@|��Bf  C���
=?�@�Q�Bk�RC'}q                                    Bx�K�  �          @�{�ff?c�
@|��Bf�CW
�ff?
=q@�Q�Bk��C'&f                                    Bx�K�:  �          @����
=q?�\)@�33BmffC�{�
=q?B�\@�{BuffC �R                                    Bx�L�  �          @�\)��R?n{@y��B_=qC^���R?
=@~{Be�C&��                                    Bx�L�  �          @��R�&ff?\(�@u�BY�
C!�
�&ff?�@x��B_
=C(��                                    Bx�L*,  �          @�
=�$z�?s33@vffBZG�C���$z�?(�@z�HB`�C&�                                    Bx�L8�  �          @����?��
@�=qBmffC���?+�@���Bt��C#                                      Bx�LGx  �          @�
=�\)?s33@x��B^33C0��\)?��@}p�Bd33C&ff                                    Bx�LV  �          @�  ��?��\@~�RBd=qC���?(��@���Bj��C$Y�                                    Bx�Ld�  �          @�\)�
=?s33@�  BfG�C)�
=?
=@��Bl�C%��                                    Bx�Lsj  �          @����?�  @��HBnffCL���?#�
@��Buz�C#��                                    Bx�L�  �          @�\)��?fff@���Bu\)C�)��?�@��RB{�C%��                                    Bx�L��  �          @�\)�p�?O\)@�33Bpz�C��p�>�ff@��Bu��C(�)                                    Bx�L�\  �          @�\)��
?@  @��BlffC!����
>Ǯ@��Bq=qC*\)                                    Bx�L�  �          @�\)���?c�
@��Blp�C�H���?�@��
Brp�C&�                                    Bx�L��  �          @��R�G�?�{@��Br��CG��G�?@  @�{Bz��C��                                    Bx�L�N  �          @�
=���?���@�=qBnG�Ch����?5@���Bv  C!��                                    Bx�L��  �          @��R��?s33@���Bn�Cff��?
=@��
Bu=qC$��                                    Bx�L�  �          @���Q�?��@��
Bw33C33��Q�?333@�ffB�C (�                                    Bx�L�@  �          @�
=��?z�H@��Bn
=C�f��?�R@�(�Bt�C$=q                                    Bx�M�  �          @�
=��?�ff@�By�RCff��?+�@�Q�B���C ��                                    Bx�M�  �          @��R��?��@�
=B}��C�3��?5@���B�G�Cu�                                    Bx�M#2  �          @�ff��=q?��@�G�B��RCJ=��=q?5@��
B��RC��                                    Bx�M1�  �          @�
=��(�?�\)@��HB��
C�
��(�?:�H@�p�B�.C��                                    Bx�M@~  �          @�
=�
�H?p��@��\Bo��C���
�H?z�@�z�BvG�C%�                                    Bx�MO$  �          @�
=�
=?h��@��
Bs\)C�f�
=?
=q@�By�C%��                                    Bx�M]�  �          @�ff�z�?n{@��
Bt��C�q�z�?\)@�{B{�C$��                                    Bx�Mlp  T          @�ff��
?k�@��
Bu  C�{��
?\)@�B{�
C$�
                                    Bx�M{  �          @��R�	��?c�
@�33Bq�C���	��?�@��Bx  C&J=                                    Bx�M��  �          @�
=��?aG�@�p�Bx
=C�\��?�\@�\)B~�\C%��                                    Bx�M�b  	          @�
=��?n{@��
Br��CaH��?\)@�Byz�C%8R                                    Bx�M�  �          @��R�Q�?aG�@��BrCs3�Q�?�@�p�By{C&T{                                    Bx�M��  �          @��R�
=?fff@��Bs\)C��
=?�@�By��C%�                                    Bx�M�T  �          @�ff��?k�@��HBr  C����?��@��Bx��C%^�                                    Bx�M��  �          @��
�H?Q�@��BqQ�C\)�
�H>�ff@��
Bw
=C(0�                                    Bx�M�  �          @�ff��
?E�@���Bx{C����
>���@�ffB}��C)
=                                    Bx�M�F  �          @��R�?Y��@�z�BuffC��>�@�ffB{�C')                                    Bx�M��  �          @�{���?^�R@��\Bq�C�����?�\@�z�Bx(�C&��                                    Bx�N�  �          @��33?k�@�33Bt��CǮ�33?\)@�p�B{�
C$޸                                    Bx�N8  �          @��Ϳ�Q�?aG�@�(�Bz�C����Q�?�\@�{B�ǮC%33                                    Bx�N*�  T          @����R?�ff@}p�Bh�C����R?0��@�G�Bo�C"Ǯ                                    Bx�N9�  �          @��G�?���@|(�Be��C���G�?8Q�@���Bmp�C"xR                                    Bx�NH*  �          @��\)?��\@~�RBiQ�C�=�\)?(��@��Bp�C#�)                                    Bx�NV�  T          @��R���?p��@��HBp��C8R���?z�@���Bwz�C$��                                    Bx�Nev  �          @�ff��?Tz�@��\Bp�C{��>�@�z�BvC'�R                                    Bx�Nt  �          @�ff�	��?Y��@��HBr{Cu��	��>�@���Bx�C'p�                                    Bx�N��  �          @����?W
=@��\Bt��C���>��@�z�BzC'T{                                    Bx�N�h  �          @����?W
=@��Br��Ch���>��@��
Bx�
C'z�                                    Bx�N�  �          @�
=�?O\)@w�Bo�C���>�@{�Bu�C'}q                                    Bx�N��  �          @�  �(�?W
=@uBjffC���(�>��H@y��BpffC'O\                                    Bx�N�Z  �          @����(�?\(�@w
=Bjp�C���(�?�\@z�HBp�\C&�                                    Bx�N�   �          @�\)���R?\(�@z�HBtffC�q���R?   @~�RB{
=C%��                                    Bx�Nڦ  �          @����(�?^�R@�=qBx�C
=��(�?   @�(�B\)C%��                                    Bx�N�L  �          @�{�
=q?n{@���Boz�C�R�
=q?\)@��
Bv=qC%�                                     Bx�N��  T          @��
�H?J=q@�=qBq�C )�
�H>��@��
Bw
=C)5�                                    Bx�O�  �          @���?(��@���Bv�\C"���>�z�@�33B{�C,33                                    Bx�O>  �          @�  ��\?.{@}p�Bu��C!����\>��R@�  Bz�HC+8R                                    Bx�O#�  �          @����33?:�H@}p�Bt�C s3�33>�p�@�Q�By�C)�f                                    Bx�O2�  �          @�  ���R?E�@}p�Bv�C�\���R>��@���B|��C(s3                                    Bx�OA0  �          @��ÿ�Q�?W
=@�  BxC�=��Q�>��@��Bz�C&T{                                    Bx�OO�  �          @�G���p�?&ff@�G�Bz��C!���p�>�\)@��\BC+�                                    Bx�O^|  �          @�  ��(�>�@���B}{C&��(�=�Q�@�G�B�  C1L�                                    Bx�Om"  �          @�{��Q�?:�H@z�HBxffC@ ��Q�>�p�@~{B~(�C)(�                                    Bx�O{�  �          @�{�z�?s33@s�
Bl�
Cff�z�?
=@xQ�Bt33C$
=                                    Bx�O�n  �          @�(��ff?k�@n�RBi�\C@ �ff?z�@s33Bp�C$��                                    Bx�O�  �          @�(���?xQ�@n�RBi�RC���?�R@s33Bq=qC#k�                                    Bx�O��  �          @�33��?h��@uBw�C� ��?��@z=qBQ�C#k�                                    Bx�O�`  �          @��\����?aG�@tz�Bx��C(�����?�@x��B��C#��                                    Bx�O�  �          @��
��ff?u@w
=Bxp�C
��ff?
=@{�B�k�C!�\                                    Bx�OӬ  �          @��H��(�?�{@tz�Bw�C+���(�?@  @z=qB���Cs3                                    Bx�O�R  �          @��
����?��@z�HB�  C&f����?8Q�@�  B�W
CJ=                                    Bx�O��  �          @�  ��ff?�p�@���B
=C����ff?Y��@��
B�u�CW
                                    Bx�O��  �          @�z��?\(�@��B�
=C��>�@�
=B���C%޸                                    Bx�PD  �          @��R���?\(�@���B�8RCLͿ��>�ff@��\B��C%�R                                    Bx�P�  m          @�=q��?h��@��
B�L�CaH��>��H@�B�.C$�\                                    Bx�P+�  �          @��\��
?aG�@���ByG�C�H��
>��@��HB�C'�                                    Bx�P:6  �          @��H��\?fff@�G�Bz  C���\?   @�33B�� C&\)                                    Bx�PH�  �          @��\�z�?k�@���Bx
=C��z�?�\@��\B{C&0�                                    Bx�PW�  �          @�33��
?^�R@���Bz33C!H��
>�@��B�k�C'��                                    Bx�Pf(  �          @�����?^�R@�{BuQ�C�f��>��@�  B{�HC'��                                    Bx�Pt�  �          @���Q�?^�R@�z�Bs�\C���Q�>�@�ffBz(�C'p�                                    Bx�P�t  �          @��
=?�  @���Bp  C�q�
=?(�@�(�BwC$�                                    Bx�P�  �          @�(��Q�?��@~{Bl�C���Q�?(��@���Bt��C"��                                    Bx�P��  �          @�G��?���@w�Bi�C���?@  @}p�Br�C E                                    Bx�P�f  �          @����\?}p�@xQ�Bn�C
=��\?(�@|��Bv��C#8R                                    Bx�P�  �          @�
=�33?���@u�Bkz�CO\�33?333@z=qBt=qC!!H                                    Bx�P̲  �          @��R�?��@r�\Bh�C���?5@xQ�BqQ�C!�                                    Bx�P�X  �          @�ff��
?�G�@s�
BkC�
��
?#�
@x��Bt
=C"�\                                    Bx�P��  �          @���Q�?�G�@uBq\)C����Q�?!G�@z�HBz  C!�R                                    Bx�P��  �          @��\��\?p��@j�HBj�C#���\?z�@o\)Bq��C$\                                    Bx�QJ  �          @����p�?+�@���B��C�\��p�>�=q@�ffB�z�C)޸                                    Bx�Q�  �          @�G���Q�?J=q@�  B�  CW
��Q�>�{@���B�{C&��                                    Bx�Q$�  T          @�p���
=?333@��B�8RC���
=>u@�ffB��3C*��                                    Bx�Q3<  �          @��Ǯ?(��@�(�B��)C�ÿǮ>L��@�B��)C,�{                                    Bx�QA�  �          @�p���=q?+�@��B�\C���=q>aG�@���B��C,=q                                    Bx�QP�  �          @�ff��p�?0��@�p�B��=C����p�>aG�@�
=B��HC+W
                                    Bx�Q_.  �          @�z΅{?!G�@��B�\)C�q��{>.{@�ffB��\C,�q                                    Bx�Qm�  �          @�33���?(�@��
B��3CO\���>��@��B���C-��                                    Bx�Q|z  �          @��H��p�?+�@�(�B�G�CxR��p�>L��@�p�B�\C*��                                    Bx�Q�   �          @��\�G�?5@|(�B�{C	�f�G�>��R@\)B�G�C8R                                    Bx�Q��  �          @����=p�?:�H@z=qB�ffCc׿=p�>���@}p�B�C��                                    Bx�Q�l  �          @{��0��?(��@s�
B���C0��0��>�=q@vffB�W
Cs3                                    Bx�Q�  �          @}p���R?.{@vffB�33C���R>�\)@y��B�(�Cz�                                    Bx�QŸ  �          @�녿
=q?&ff@~{B��
C���
=q>�  @���B��C\)                                    Bx�Q�^  �          @vff�L��?:�H@qG�B�k�Bҽq�L��>�{@tz�B�B�B���                                    Bx�Q�  
�          @z�H�L��?\)@w�B��=B�{�L��>#�
@z=qB�{B�(�                                    Bx�Q�  �          @\)=���>���@~{B�
=B�B�=���    @\)B�=qC��f                                    Bx�R P  �          @�33���R>��@�=qB�� B�#׾��R�#�
@��HB�B�C5#�                                    Bx�R�  �          @�  ��z�?G�@���B���CG���z�>���@��\B���C$�                                    Bx�R�  �          @���
=?W
=@�{B�C�
��
=>���@�  B�u�C!B�                                    Bx�R,B  �          @�{��33?Y��@�ffB�L�C}q��33>��@�Q�B���C @                                     Bx�R:�  �          @��H���?E�@��
B��C�=���>�{@�B�p�C"�\                                    Bx�RI�  �          @�녿�33?p��@�G�B���C�q��33>�@��
B�CL�                                    Bx�RX4  �          @��
���?�z�@���B�aHC
n���?5@��
B�L�C�\                                    Bx�Rf�  �          @��Ϳ�G�?���@��HB�B�C�׿�G�?(�@�B��HC&f                                    Bx�Ru�  �          @�  ���?�=q@�p�B�  C����?#�
@�Q�B���C��                                    Bx�R�&  �          @������?���@��RB���C������?=p�@�=qB�(�C�)                                    Bx�R��  
�          @����˅?�z�@��
B��=C�ÿ˅?0��@�
=B���C�
                                    Bx�R�r  �          @�33����?��R@��B���CT{����?B�\@���B�C�=                                    Bx�R�  �          @����(�?���@�G�B�\C	aH��(�?c�
@��B�p�C�)                                    Bx�R��  �          @������?�p�@���B�ǮC
�=���?G�@�Q�B��C                                    Bx�R�d  �          @�  ��
=?�(�@��B�aHC�ÿ�
=?=p�@�
=B�B�C��                                    Bx�R�
  �          @�녿�G�?�p�@���B�L�C�=��G�?@  @�Q�B�
=C��                                    Bx�R�  �          @�G���
=?�G�@���B�33C
����
=?G�@�Q�B�Q�C��                                    Bx�R�V  �          @������?��@��B���C�����?O\)@�B�C�                                    Bx�S�  �          @�������?�p�@��B��
CͿ���?E�@�33B�G�C��                                    Bx�S�  �          @�\)�
=q?��R@�Q�B��B�녿
=q?L��@��
B��3B�{                                    Bx�S%H  �          @���#�
?�33@|(�B���B��#�
?8Q�@���B�G�CǮ                                    Bx�S3�  �          @�G��}p�?��@���B���CͿ}p�?0��@�(�B�p�C33                                    Bx�SB�  T          @��Ϳ�\)?��@��B�C)��\)?
=@��\B�ǮC {                                    Bx�SQ:  �          @��R��=q?��R@�G�B�B�CǮ��=q?:�H@���B��C�                                    Bx�S_�  �          @�{���?�33@���B�(�C#׿��?#�
@���B��CW
                                    Bx�Sn�  �          @���ff?�33@�G�B���C�Ϳ�ff?!G�@�z�B�G�CǮ                                    Bx�S},  �          @�(����H?�33@���B�\)C����H?#�
@��
B�Ch�                                    Bx�S��  �          @�(���p�?���@�Q�B�C�H��p�?�R@�33B��CJ=                                    Bx�S�x  �          @�����?�=q@���B�u�Cn����?�@�z�B�ffC 8R                                    Bx�S�  �          @�Q����?�=q@�z�B�W
Cn����?��@�\)B�G�C �f                                    Bx�S��  �          @��H����?�\)@�  B��RC0�����?z�@��HB�B�C.                                    Bx�S�j  �          @������?��@�Q�B�C�\����?
=@��B�aHC�{                                    Bx�S�  �          @�(�����?�@���B��RC�H����?(�@���B��qC.                                    Bx�S�  �          @�(���ff?��\@���B��C�3��ff>��@��B��C#G�                                    Bx�S�\  �          @�����{?�  @�  B��
C�H��{>�ff@��\B���C!�3                                    Bx�T  �          @�z�u?�{@��B�
=C�R�u?�@�Q�B���CQ�                                    Bx�T�  �          @��Ϳ��\?��
@�p�B�k�C����\>��H@�Q�B�=qC��                                    Bx�TN  �          @�����?��@��B�G�C  ���?
=q@�  B�.C}q                                    Bx�T,�  �          @�p�����?�{@�p�B�� CͿ���?��@���B���C�
                                    Bx�T;�  �          @�ff���?}p�@��B�33Cn���>�(�@�=qB���C�                                    Bx�TJ@  �          @�z��>��
@��B�aHC*!H���8Q�@�  B��fC9��                                    Bx�TX�  �          @���ٙ�>��
@���B��C)c׿ٙ��B�\@���B�u�C:G�                                    Bx�Tg�  �          @�G���z�>��@��RB��fC&33��zὸQ�@�\)B�(�C7=q                                    Bx�Tv2  �          @��׿˅>�G�@�ffB�#�C$��˅�u@�
=B��qC65�                                    Bx�T��  �          @�녿У�?�@�\)B��HC!���У�<��
@�Q�B�(�C3�                                    Bx�T�~  �          @������>�(�@���B�ǮC$녿��ͽ�Q�@�=qB�=qC7
=                                    Bx�T�$  �          @�{��(�>�z�@�{B��RC(�=��(��u@�{B��C=k�                                    Bx�T��  �          @�Q쿼(�?E�@���B���CO\��(�>u@��RB�\C*�)                                    Bx�T�p  T          @�����(�?���@�{B�#�C
��(�?\)@�G�B��fC8R                                    Bx�T�  �          @�녿��
?:�H@�  B��3CxR���
>8Q�@���B���C-O\                                    Bx�Tܼ  �          @�녿�?\)@��B��fC�\��    @�33B��\C4�                                    Bx�T�b  �          @��
��z�?&ff@��B�ǮC:ῴz�=��
@��B�=qC0��                                    Bx�T�  �          @��H���?0��@��\B���C�����=�@�(�B�� C/�                                    Bx�U�  �          @�=q��\)?z�@��HB�.C쿯\)<#�
@�(�B��C3�f                                    Bx�UT  �          @��\��33?!G�@��\B���C�\��33=u@��
B�#�C1�=                                    Bx�U%�  �          @�33��  ?5@��B��C�q��  >\)@��B��qC.�                                     Bx�U4�  �          @�����?@  @���B���C�q���>8Q�@��B��fC-n                                    Bx�UCF  �          @�p���{?8Q�@��HB�\C���{>\)@���B��C/#�                                    Bx�UQ�  �          @��ÿ�?#�
@�33B�� C!���=L��@�z�B�C2c�                                    Bx�U`�  �          @�  ��?�R@��\B�k�C"&f��=#�
@��
B��
C2��                                    Bx�Uo8  �          @�
=��p�?\)@���B�#�C$8R��p����
@��B�{C4�                                    Bx�U}�  T          @�\)����?z�@��\B�#�C"�����ͼ#�
@��
B�W
C45�                                    Bx�U��  �          @����(�>��H@���B�C&{��(�����@��\B�u�C6��                                    Bx�U�*  �          @�{���R?!G�@�
=B�C"� ���R=L��@���B�u�C2�{                                    Bx�U��  �          @�����?k�@���B�k�C�����>��R@�\)B���C&p�                                    Bx�U�v  �          @��Ϳ���?p��@��
B�L�CzῨ��>�{@�ffB��C%��                                    Bx�U�  �          @��\����?z�H@��HB�u�C
5ÿ���>\@�B�(�C �q                                    Bx�U��  �          @���#�
?���@��
B�ǮB�(��#�
?s33@�G�B�\B�aH                                    Bx�U�h  �          @��\�8Q�?�  @�33B�.B�3�8Q�?#�
@�\)B���C
B�                                    Bx�U�  �          @�33���?��R@�=qB�.B�
=���?aG�@�\)B��B���                                    Bx�V�  �          @��ÿJ=q?���@�Q�B�\B��)�J=q?=p�@���B�
=C�q                                    Bx�VZ  �          @�33�L��?���@��\B��)B�p��L��?E�@�\)B�\C��                                    Bx�V   �          @�\)�=p�?���@��RB���B���=p�?G�@��B�(�Cu�                                    Bx�V-�  T          @�Q�O\)?�@��B���B�\�O\)?@  @�(�B���C	(�                                    Bx�V<L  T          @�(��O\)?��
@���B���B��O\)?�R@���B��C�H                                    Bx�VJ�  �          @�ff���?�ff@���B�G�CE���?\(�@��RB���C޸                                    Bx�VY�  �          @��R��33?��@�p�B�Q�C�Ϳ�33?!G�@���B�B�C!H                                    Bx�Vh>  �          @����ff?��\@���B�u�C�׿�ff?\)@���B�� C�)                                    Bx�Vv�  �          @�\)����?s33@���B�8RC
n����>��@��B���C&h�                                    Bx�V��  �          @��׿�33?
=@��B���C�)��33��G�@�z�B��C9�{                                    Bx�V�0  �          @��
����?k�@�p�B�  C�)����>W
=@�  B�  C)8R                                    Bx�V��  �          @�=q�z�H?z�H@�(�B���C��z�H>��@�
=B�C$�H                                    Bx�V�|  �          @�33�n{?���@���B�ǮC0��n{>\@�Q�B�aHC��                                    Bx�V�"  �          @��ÿO\)?��@��B��3B�녿O\)>�{@��RB�ǮC)                                    Bx�V��  
�          @�z῅�?�\)@�p�B�{C���>Ǯ@���B�33C�
                                    Bx�V�n  T          @������?�@��
B��{C^�����>�ff@�\)B�
=CY�                                    Bx�V�  �          @����?���@��B�k�C޸��>�p�@���B��qC"��                                    Bx�V��  �          @��ͿaG�?��
@��B�z�B�aH�aG�?
=q@�G�B��
C��                                    Bx�W	`  �          @�{��ff?�(�@��
B�8RB��ῆff?:�H@���B���C�                                    Bx�W  �          @����ff?���@�(�B��\C � ��ff?z�@�Q�B�L�C�                                    Bx�W&�  �          @���p�?��@���B��=B�����p�?h��@��RB�33Cu�                                    Bx�W5R  �          @�
=���\?\@�33B��=C\���\?E�@�Q�B�aHC�=                                    Bx�WC�  �          @����\)?�\)@�=qB���C5ÿ�\)?aG�@�  B���Cc�                                    Bx�WR�  �          @�G���p�?У�@��
B�ffB��ÿ�p�?k�@���B�33CO\                                    Bx�WaD  �          @��
���
?\@��B��{CͿ��
?G�@��B��{C�                                     Bx�Wo�  �          @�(���G�?��@��B���CaH��G�?L��@��B���C                                    Bx�W~�  �          @�(����
?ٙ�@�B�W
B��ÿ��
?u@�(�B�Q�C�                                    Bx�W�6  T          @�녿���?�33@���B�\)C!H����>��@���B�Q�C!L�                                    Bx�W��  T          @�ff��ff?��@��\B��C(���ff?+�@�\)B�8RC�                                    Bx�W��  T          @�G����?�p�@��RB�(�C�쿧�?�@��HB�p�C:�                                    Bx�W�(  �          @�ff����?�z�@��B�ǮCJ=����>��@�\)B�k�C ��                                    Bx�W��  �          @��Ϳ���?}p�@��B�(�C	B�����>�z�@�Q�B��C$�                                    Bx�W�t  �          @��þ���G�@�ffB���CA^����O\)@�(�B�W
Cp��                                    Bx�W�  �          @�  �O\)?Q�@��B��
C�3�O\)=�@�B�{C+s3                                    Bx�W��  �          @�33�s33?u@�p�B�W
C���s33>W
=@�Q�B�{C'��                                    Bx�Xf  �          @�\)�W
=?���@���B��B�
=�W
=>���@�z�B�� C�{                                    Bx�X  �          @�����  ?��@��\B�  C����  >�{@�{B��)C!�                                    Bx�X�  �          @�33���\?n{@��B��\C	�쿂�\>.{@�  B���C*L�                                    Bx�X.X  �          @����ff?!G�@�ffB�=qC:ῆff�\)@��B�  C;u�                                    Bx�X<�  �          @�\)��ff>�G�@��HB��C#׿�ff���R@�33B���CD5�                                    Bx�XK�  T          @�Q�+�>��@�{B�
=C�)�+����
@�ffB�  CM��                                    Bx�XZJ  �          @����  ��
=@�z�B��{CF�3��  ���@���B���C\8R                                    Bx�Xh�  �          @������<��
@���B�L�C3
=��녿!G�@�33B��
CP                                    Bx�Xw�  �          @�녿�{?��@�z�B���CǮ��{�\)@�B��C:�3                                    Bx�X�<  �          @�������>�p�@�(�B���C#  ������Q�@�(�B���CD�R                                    Bx�X��  �          @��׿�p�?���@��B���C\)��p�>�
=@�{B�W
C&5�                                    Bx�X��  �          @�=q�(�?�{@�z�Bs��C!H�(�?(�@���B�33C$W
                                    Bx�X�.  �          @�z���?�33@�
=Bt�
C=q��?#�
@�(�B�{C#�                                    Bx�X��  �          @�\)�
=?�p�@��\Bwz�C���
=?333@�  B�
=C!��                                    Bx�X�z  �          @�33���?�
=@��
Bq�C����?+�@�G�B���C#�                                    Bx�X�   �          @���{>\@j=qBip�C*B��{�W
=@j�HBj�HC9c�                                    Bx�X��  �          @����G�>��@g�BeffC(c��G���@i��Bh33C7\                                    Bx�X�l  �          @��z�?��@n�RBe�RC&�
�z�u@qG�Bi��C5h�                                    Bx�Y
  T          @��R��>Ǯ@s�
Bk
=C*33�녾k�@tz�Bl\)C9�                                     Bx�Y�  �          @�33�z�?�Q�@�Q�B^��C
�z�?E�@�{Bo
=C!�{                                    Bx�Y'^  �          @�Q���?�\)@z�HB]��C����?5@��HBm{C#�                                    Bx�Y6  �          @�(����?��R@uB`��C8R���?
=@\)Bn�RC%G�                                    Bx�YD�  �          @�z��?��\@w�Bbp�Cc��>�p�@\)Bm33C*�q                                    Bx�YSP  �          @������?�  @vffB_��CW
���>�33@}p�Bj{C+�f                                    Bx�Ya�  �          @�G��!�?z�H@|��B^�C�!�>��R@��Bg�\C,�                                    Bx�Yp�  �          @�33�%?O\)@l(�BV�\C"���%>8Q�@qG�B]�RC0�                                    Bx�YB  �          @������?xQ�@r�\Bd{C���>��
@y��Bn��C+޸                                    Bx�Y��  �          @��\���?�33@w�Bg��C�\���>�@�Q�Bup�C'^�                                    Bx�Y��  �          @���Q�?h��@q�B`
=C\�Q�>��@xQ�Bi=qC-�                                     Bx�Y�4  �          @���(�?@  @VffBR��C"���(�>.{@Z�HBY��C0�                                    Bx�Y��  �          @�33���c�
@S33BY��CJ=q����Q�@FffBGffCUk�                                    Bx�YȀ  �          @6ff��33�n{@��Bk��C[���33���@��BP{CeL�                                    Bx�Y�&  �          @Y����G���\)@8Q�Bm�\Ci�)��G���@&ffBL=qCqE                                    Bx�Y��  �          @��H��\)��R@u�Bc��Csn��\)�7
=@X��B?�\Cx��                                    Bx�Y�r  �          @����{��z�@w�Bt{C]#׿�{��
@dz�BW{Cg�H                                    Bx�Z  �          @��׿Ǯ>�@��RB�C#���Ǯ��\)@�\)B��HC>�                                    Bx�Z�  �          @�z��G�>���@���B��{C*h���G���ff@�Q�B��qCBO\                                    Bx�Z d  �          @��׿�>���@���B���C)�����ff@�z�B��CA�
                                    Bx�Z/
  �          @��ÿ�
=>Ǯ@�{B�33C&�)��
=��33@�{B�u�C?��                                    Bx�Z=�  �          @��׿�>.{@c33Bz  C.����@aG�Bv�HCA�q                                    Bx�ZLV  �          @Y���O\)?�ff@Dz�B�ffB���O\)>��H@Mp�B�#�C�                                    Bx�ZZ�  �          @p�׿�{>��
@S33B~(�C(��{��\)@S33B~�C=��                                    Bx�Zi�  "          @i����G�=��
@J=qBs��C1c׿�G���@HQ�Bn��CCaH                                    Bx�ZxH  �          @k��޸R?�\@L(�Bq��C#� �޸R��\)@N�RBv��C6T{                                    Bx�Z��  �          @\(���=q?W
=@9��Bi(�C녿�=q>��R@@  Bw33C(�                                    Bx�Z��  �          @r�\����?Y��@FffB^
=C\)����>�\)@Mp�Bi�\C+Ǯ                                    Bx�Z�:  �          @l(��
=?�33@0��BC�C��
=?!G�@:�HBSC#k�                                    Bx�Z��  �          @r�\�(�?�=q@*�HB4=qC!H�(�?z�@4z�BA��C&��                                    Bx�Z��  �          @i����R?B�\@{B.=qC"�3��R>���@$z�B7  C-#�                                    Bx�Z�,  �          @s�
�*=q?
=q@)��B1G�C(���*=q=#�
@-p�B5�
C3�                                    Bx�Z��  �          @y���1G�?   @+�B.�C)ٚ�1G�    @.�RB2=qC4�                                    Bx�Z�x  �          @}p��4z�?xQ�@'
=B$�C!
=�4z�>��@/\)B/ffC*�                                    Bx�Z�  �          @|���5�?��\@$z�B!�
C E�5�?�@-p�B-
=C)�                                    Bx�[
�  �          @s�
�\)?�Q�@(��B.��C���\)?.{@3�
B=C$ٚ                                    Bx�[j  �          @n�R��{?��@;�BPC�R��{?=p�@HQ�BeQ�Ck�                                    Bx�[(  �          @x�ÿ�\)?�  @HQ�BYQ�CLͿ�\)?#�
@S�
Bl��C!�                                    Bx�[6�  �          @\)�{?�@6ffB6�C�{?�{@G
=BM��CxR                                    Bx�[E\  �          @�{�
=q?�@A�B9Q�C
O\�
=q?���@UBS\)C��                                    Bx�[T  �          @���(�@G�@&ffB��C	��(�?޸R@>�RB3��C}q                                    Bx�[b�  �          @��\�#33@$z�@�B  Cٚ�#33@z�@4z�B"ffC��                                    Bx�[qN  �          @��!G�@"�\@$z�B
=C�q�!G�@   @@��B,z�C}q                                    Bx�[�  �          @��\�'
=@&ff@-p�B33C+��'
=@�@I��B.p�C�                                    Bx�[��  �          @���*�H@5�@
=A���C^��*�H@z�@6ffB�\C
�q                                    Bx�[�@  �          @�33�8Q�@-p�@A�G�C�8Q�@p�@4z�Bz�C�=                                    Bx�[��  �          @�Q��G�@2�\@��A�z�C
+��G�@33@0  Bz�C�\                                    Bx�[��  �          @��\�P  @9��@AͮC
E�P  @�@'�BG�C!H                                    Bx�[�2  �          @����.{@��@�A�RC��.{?�G�@�HB�
C
=                                    Bx�[��  T          @y�����@�@�
Bp�C�����?��@*=qB,ffC8R                                    Bx�[�~  T          @}p���R?�Q�@Q�B�\C����R?�Q�@.{B.{C�f                                    Bx�[�$  �          @z=q�#�
?��
@Q�B	=qC:��#�
?���@(�B ��C��                                    Bx�\�  �          @z�H�ff@Q�� ��� 
=C�)�ff@%���\���C)                                    Bx�\p  �          @~{��H@(��{��C	�)��H@%�޸R��p�C
                                    Bx�\!  �          @r�\�Q�@���Q���{C	��Q�@!G���(���p�CW
                                    Bx�\/�  �          @~�R�!�@=q��z����Ck��!�@/\)�����G�C��                                    Bx�\>b  �          @|(��&ff@(���
=��=qC���&ff@!녿�����{C��                                    Bx�\M  �          @i���Dz�?���\��C�Dz�?�z���33C!H                                    Bx�\[�  �          @���:�H?�\)@�B�\C��:�H?�\)@'
=B\)C�                                    Bx�\jT  �          @�\)��?��@�33Bs(�Cn��>W
=@�\)B�p�C.+�                                    Bx�\x�  �          @��ÿ��>��@�(�B��=C&
=��녿�@�(�B�#�CC�=                                    Bx�\��  �          @�
=���?�@��B��C$����녿
=q@�33B���CC��                                    Bx�\�F  
�          @����H>�\)@�z�B�ǮC*�f���H�G�@��HB��CL��                                    Bx�\��  �          @��H���H>��H@�33B�z�C!p����H��@��HB��)CIh�                                    Bx�\��  �          @�녿�׾�=q@�{B��fC<���׿��
@���B�CVB�                                    Bx�\�8  �          @vff�B�\?�33@e�B�B�=q�B�\>\@o\)B�33CT{                                    Bx�\��  �          @�������?�R@~{B�u�C�׿��׾aG�@�Q�B��RC=@                                     Bx�\߄  �          @�ff��z�>�@�33B��3C$p���z��@�33B�CCJ=                                    Bx�\�*  �          @��R��
=>�ff@���B�C���
=��\@���B�=qCK5�                                    Bx�\��  �          @�
=��p�����@���B�B�C?aH��p����@��HBo�
CT��                                    Bx�]v  �          @�zῴz��G�@�p�B�ffC8�f��zῌ��@�G�B�#�CY�R                                    Bx�]  �          @�zῑ녿��@���B��Ca5ÿ�녿�(�@mp�Bi��Cp{                                    Bx�](�  �          @���#�
�L��@y��B�� Cgs3�#�
��{@h��B�
=Cx\)                                    Bx�]7h  �          @2�\���׿�{?�z�B,\)CfT{���׿�
=?�=qB�
Cl#�                                    Bx�]F  �          @C33�\�Ǯ���B33C�8R�\�����ff�qC{��                                    Bx�]T�  
�          @;�?O\)>L���*=qB�A[\)?O\)?B�\�#�
��B-z�                                    Bx�]cZ  �          @_\)>��R�.{�\(�¨33C�>��R?z��Y��\Bv
=                                    Bx�]r   �          @vff?�{����g��fC�� ?�{>\�g�B�A�p�                                    Bx�]��  �          @r�\>��=u�Z=q¦�A  >��?L���Tz��RB{
=                                    Bx�]�L  �          @U��ff@(�?��B  B�aH��ff?�p�@�
B3z�C�                                    Bx�]��  �          @dz῜(�@��@��B3G�B�.��(�?Ǯ@7
=B]\)C �                                    Bx�]��  �          @����=q@0��@>{B,(�B�ff��=q@33@`��BVz�B�8R                                    Bx�]�>  �          @�G����@A�@r�\BC��B�=q���@�@��BqG�B�3                                    Bx�]��  �          @�녿L��@Dz�@dz�BAp�B�aH�L��@��@��Br(�B�\                                    Bx�]؊  �          @��R�Q�@<��@e�BE�RB�.�Q�@@�z�BvQ�B�{                                    Bx�]�0  �          @�����G�@"�\@|(�B]  B�8R��G�?���@��B��B�G�                                    Bx�]��  �          @�(��k�@�@��RBo
=B�  �k�?��
@��
B�B�B�\)                                    Bx�^|  �          @��Ϳ�(�@)��@�  B^=qB�W
��(�?У�@�\)B�� B�p�                                    Bx�^"  �          @�33���
@#33@�\)B`(�B�ff���
?��
@�{B��C��                                    Bx�^!�  �          @�p���@6ff@uBS�\B��ÿ�?�@�(�B�.B���                                    Bx�^0n  �          @�Q�>\@,(�@fffBS�RB�#�>\?�@��B���B��{                                    Bx�^?  �          @�33��@*�H@l��BV�
Bȅ��?�G�@�ffB�  Bң�                                    Bx�^M�  �          @��R��ff@P  @VffB6G�B�녾�ff@��@�  Bj(�B�ff                                    Bx�^\`  �          @w
=��
=?��@J=qB��\B��
��
=?!G�@X��B��B�u�                                    Bx�^k  �          @��ͽ��
�.{@��HB�C��R���
��\@��B���C��q                                    Bx�^y�  �          @�
=�.{�k�@�{B���CjaH�.{���@�Q�B�u�C�T{                                    Bx�^�R  �          @��
�#�
�aG�@��HB��C��ý#�
���H@�B��\C�\                                    Bx�^��  �          @�    >���@��B�=qB��    �:�H@��B�C��q                                    Bx�^��  n          @��;��?�\@��B��RC#׾�׿�\@��B��Cb�3                                    Bx�^�D  :          @�  �   ?z�H@��
B���B���   ��\)@�
=B�(�C;�=                                    Bx�^��  �          @�=q��G�?�=q@s�
B���B�uþ�G�>��@���B�{C	G�                                    Bx�^ѐ  �          @o\)��\?�
=@\(�B�  B�Ǯ��\>�{@g�B��qC8R                                    Bx�^�6  �          @�33��(�@*=q@Z=qBNp�B�p���(�?��@|(�B��fB�.                                    Bx�^��  �          @��!G�@7�@aG�BHp�B̊=�!G�?��H@�33B}33B�Q�                                    Bx�^��  �          @�\)���?�@��RB}p�B��f���?J=q@���B���C��                                    Bx�_(  �          @��ÿ��H?s33@�  B�Q�C�R���H�\)@�33B�k�C:��                                    Bx�_�  �          @�33���׼#�
@�z�B��HC45ÿ��׿�{@�  B�{CZ��                                    Bx�_)t  �          @��Ϳ�ff��\)@��RB�p�C@!H��ff��33@�Q�B�{Cc                                    Bx�_8  �          @�����R��@���B�aHCILͿ��R�Ǯ@���B�G�Cg�                                     Bx�_F�  �          @�zῆff>�  @�Q�B�8RC&�ÿ�ff�W
=@�{B�L�CZ�                                    Bx�_Uf  �          @P�׿&ff?(��@>{B�� C�H�&ff���
@B�\B�ǮC:��                                    Bx�_d  �          @QG����R��(�@+�Bo�HCD:῾�R����@   BV�CW��                                    Bx�_r�  �          @0�׿n{��?�AQG�CuJ=�n{�<#�
>�\)Cu��                                    Bx�_�X  �          @x�ýu�R�\��33��=qC�� �u�.�R�(���0p�C�ff                                    Bx�_��  �          @�\)��Q��H���G
=�3  C�"���Q����q��kz�C�Ф                                    Bx�_��  �          @�33��Q��e?�AC�J=��Q��g
=��p���=qC�J=                                    Bx�_�J  �          @��
��G��|��@G�A��C�0���G���(�?�p�Az�RC�E                                    Bx�_��  �          @�{����z�?��As�
C�O\����=q=�Q�?��C�g�                                    Bx�_ʖ  �          @�=q�8Q������z��_�C�޸�8Q���\)�G���(�C��H                                    Bx�_�<  �          @�33��\)�����:�H�Q�C��H��\)���R���H��
=C���                                    Bx�_��  �          @�=q=u���H����
=C�S3=u���H�%���Q�C�]q                                    Bx�_��  �          @�=q>�p���G��.{����C�U�>�p��n{�qG��4�HC�ٚ                                    Bx�`.  �          @�z�?���G��a��#\)C��?��A���p��\��C�Q�                                    Bx�`�  �          @�=q>�(������q��"\)C��>�(��S33�����\(�C��                                    Bx�`"z  T          @��?�\��ff�tz��!��C�AH?�\�Vff��=q�[��C�Q�                                    Bx�`1   �          @�
=?O\)�j=q����C�C�C�?O\)�p����H�|33C�*=                                    Bx�`?�  �          @��?J=q�o\)��33�<
=C�  ?J=q�%�����t��C���                                    Bx�`Nl  �          @�\)?�=q?�ff�����BV(�?�=q@z�O\)��ffBeff                                    Bx�`]  �          @��?�z�@��@��A��B�?�z�@}p�@P  B  B�W
                                    Bx�`k�  �          @�33?�=q@��\@G
=B �B��)?�=q@w�@��B:
=B��                                    Bx�`z^  �          @��?�33@��@S33A��B���?�33@��\@�(�B0(�B���                                    Bx�`�  �          @��
?�{@��@P  B B��q?�{@z=q@��B8��B�G�                                    Bx�`��  �          @�(�?���@vff@�  B/�B���?���@,(�@��Be�Bn�                                    Bx�`�P  �          @��
?��\@�(�@j�HB��B���?��\@��@�  B;B�p�                                    Bx�`��  �          @޸R?���@�33@��B!{B�=q?���@p��@��BZ33B���                                    Bx�`Ü  �          @��H?���@��@�(�B�B���?���@��@��
BO(�B�(�                                    Bx�`�B  �          @�?��@��@��B�B��?��@u�@�33BW�B�.                                    Bx�`��  �          @޸R?���@�
=@��
B��B�p�?���@y��@��HBV
=B�\                                    Bx�`�  �          @��H?\@���@��RBB���?\@|(�@�{BU�\B���                                    Bx�`�4  �          @߮?���@�33@�p�B�B�G�?���@��@�BM
=B��q                                    Bx�a�  �          @��
?u@�Q�@\)B��B�B�?u@�Q�@���BJ
=B���                                    Bx�a�  �          @ٙ�?s33@���@w
=B
Q�B��?s33@���@�BF��B�(�                                    Bx�a*&  �          @�33?Tz�@���@�Q�BG�B�=q?Tz�@}p�@�Q�BV�B�u�                                    Bx�a8�  �          @��H?�{@�{@x��BffB�\)?�{@�{@�Q�B@G�B�(�                                    Bx�aGr  �          @�G�@�@�  @p�A���B��@�@�ff@j=qB=qB���                                    Bx�aV  �          @���@Z=q@����(��'
=Bk�@Z=q@���>�{@:�HBm��                                    Bx�ad�  �          @��H@`  @�33�5��\)Bp�R@`  @�=q?u@���Bp(�                                    Bx�asd  �          @�\@Q�@Ǯ�W
=��(�By  @Q�@�=q?�p�AAp�BvQ�                                    Bx�a�
  �          @�33@HQ�@��
<#�
=�\)Bff@HQ�@�(�?�p�Aa��B{�                                    Bx�a��  T          @��@W�@ə�������HBwff@W�@Å?ǮAI�Bt\)                                    Bx�a�V  �          @�33@^{@���z���p�Br33@^{@\?�\)A��Bq                                      Bx�a��  �          @��H@U@��u��Q�Bv=q@U@�ff?B�\@��Bv��                                    Bx�a��  �          @�\@h��@�ff�����p��Be��@h��@�\)�B�\�˅Bj�\                                    Bx�a�H  �          @��
@y��@�ff�7
=���BTp�@y��@��ÿ�=q�,(�B_��                                    Bx�a��  �          @�G�@���@�(��<����\)B<�H@���@�Q��  �=�BJz�                                    Bx�a�  �          @��@�33@�Q��#�
���HB=�
@�33@��׿�=q�z�BH�H                                    Bx�a�:  �          @�
=@�
=@�G���Q��z�HBG�@�
=@��
����P  BM��                                    Bx�b�  �          @�(�@���@�G����lQ�BI(�@���@��H��\)�\)BO=q                                    Bx�b�  �          @��@�ff@��\���
=BC=q@�ff@��ÿW
=�׮BL�
                                    Bx�b#,  �          @�  @�p�@���L(�����BHp�@�p�@����UBV�
                                    Bx�b1�  �          @�Q�@���@���P  ��33BB�\@���@�=q��G��ap�BQ��                                    Bx�b@x  T          @�=q@��
@��*�H��=qBGff@��
@�
=��\)��BR�                                    Bx�bO  �          @��
@���@�p��Ǯ�E�B_(�@���@��
>#�
?�G�Bb�                                    Bx�b]�  �          @��@��@��
�#�
����BR�@��@��
�^�R����B[Q�                                    Bx�blj  �          @��
@y��@��
�z�����Be��@y��@��ÿ   �n�RBl�                                    Bx�b{  �          @�@{�@�  �#33���\Bc
=@{�@�
=�@  ����Bj�                                    Bx�b��  �          @��R@�
=@�=q�2�\��p�BXG�@�
=@��
���\��33Ba��                                    Bx�b�\  �          @�Q�@�
=@���X����=qB1G�@�
=@��
��\)�`(�BB                                      Bx�b�  �          @�G�@�z�@���`  ��Q�B4z�@�z�@�ff�����iG�BE�\                                    Bx�b��  �          @��@���@�(��P  ��p�BP�@���@ʏ\��  �0Q�B\��                                    Bx�b�N  �          @��
@��\@�z��1G�����Ba��@��\@��c�
��  BjG�                                    Bx�b��  �          @�p�@�
=@���0  ��=qBJ=q@�
=@�G��z�H��BTff                                    Bx�b�  �          Ap�@�{@�33�'����HBG33@�{@˅�Q���=qBPz�                                    Bx�b�@  �          Aff@�z�@����A����HB'
=@�z�@�{�����$  B5(�                                    Bx�b��  �          A��@�
=@�{�A�����B(��@�
=@�33��33��B6��                                    Bx�c�  �          A�R@�G�@��\�=p���\)B4��@�G�@�ff��(��p�B@�H                                    Bx�c2  �          A33@�p�@Å����*�RBA�@�p�@�G�>��R@
=BE                                      Bx�c*�  �          A��@��
@Ǯ��G��"�RB?�@��
@��>\@#33BB��                                    Bx�c9~  �          A	G�@�G�@�녿����C�
B9G�@�G�@ʏ\<�>k�B>33                                    Bx�cH$  �          A
�\@�G�@���AG���\)B;G�@�G�@��ÿ�33���BF�R                                    Bx�cV�  �          A
�\@��@�ff�i����\)B:
=@��@У׿��Ap�BI\)                                    Bx�cep  �          A��@�Q�@��R�x����\)B-�\@�Q�@��
���m��B@{                                    Bx�ct  �          Aff@��@��������\B�
@��@����"�\���\B)��                                    Bx�c��  �          A�@��@�����  �=qB  @��@���A���=qB/��                                    Bx�c�b  �          A�
@�  @U���  �+�Bz�@�  @�z���Q�� �\B+                                    Bx�c�  �          A��@�z�@\)��  �R�A��@�z�@��������+z�B'
=                                    Bx�c��  �          A	G�@�(�@>�R��Q��>�\A�G�@�(�@��R���H���B0
=                                    Bx�c�T  �          A��@�=q@��R��p����RB  @�=q@��\�8Q����B/=q                                    Bx�c��  �          AQ�@�Q�@�(����R��G�B(�@�Q�@����<(���{B/33                                    Bx�cڠ  �          A�H@��@�����R�33B�@��@����L������B4��                                    Bx�c�F  �          A�@��H@��H��{���B33@��H@��
�Y����\)B9��                                    Bx�c��  �          A	G�@�(�@�p���
=��HB{@�(�@����n{��\)B7�\                                    Bx�d�  �          A	��@���@xQ������ =qB�H@���@��
���H��B5�                                    Bx�d8  �          A
�\@��
@�����\��HB{@��
@�Q��e���p�B1��                                    Bx�d#�  �          A�@�Q�@�ff���\��B"@�Q�@����'����B9\)                                    Bx�d2�  �          A33@��@�  �w
=��p�B)�R@��@�p����]�B<\)                                    Bx�dA*  �          A(�@�ff@�������G�Bff@�ff@����B�\���B+�R                                    Bx�dO�  �          A��@�\)@��R���\� �Bff@�\)@�
=�N�R��G�B*Q�                                    Bx�d^v  �          A�R@�ff@�Q����H��{B�@�ff@��
�0  ���B��                                    Bx�dm  T          A33@أ�@��������G�B=q@أ�@�33�-p���\)B\)                                    Bx�d{�  T          A��@޸R@tz���G����RA�=q@޸R@���XQ�����B33                                    Bx�d�h  �          AQ�@�G�@�(����
��G�A�33@�G�@�{�#�
��z�B                                    Bx�d�  �          A�@ۅ@�  ��z���33B(�@ۅ@�z��1���
=B                                    Bx�d��  �          Aff@ڏ\@�
=��z����B
z�@ڏ\@�33�-p���Q�B"=q                                    Bx�d�Z  �          A\)@�{@�G�����ظRB
��@�{@����'
=��B!�                                    Bx�d�   �          A�@ڏ\@�ff��=q����B��@ڏ\@����#�
�z�\B&
=                                    Bx�dӦ  �          A(�@أ�@���������p�B�R@أ�@�=q�0����G�B'��                                    Bx�d�L  �          A��@�z�@����33����B�@�z�@���{�o�B/�
                                    Bx�d��  �          Ap�@�\)@��R�����B  @�\)@ʏ\�$z��xz�B-�                                    Bx�d��  �          A�\@�G�@�  ���R�ۙ�B�
@�G�@�(��%��w�B,��                                    Bx�e>  �          A�R@�Q�@�\)������G�B�H@�Q�@�(��)���~{B-�                                    Bx�e�  �          A��@�@�������\)B�
@�@�  �$z��r=qB,                                    Bx�e+�  
�          A\)@�p�@�������z�B33@�p�@�{�   �h��B'�R                                    Bx�e:0  �          A!��@�
=@����G��ծBp�@�
=@�=q�%�m��B)G�                                    Bx�eH�  �          A"�\@�ff@��
���R��Q�B��@�ff@�  �(��^�\B,��                                    Bx�eW|  �          A#33@�{@����G���=qB��@�{@���3�
��=qB+=q                                    Bx�ef"  �          A#�
@�@�������z�B�@�@�ff�Q���\)B&��                                    Bx�et�  �          A$Q�@�=q@�����
��p�B
@�=q@�{�N�R����B%\)                                    Bx�e�n  �          A%�@�(�@���������B	��@�(�@�{�P����\)B$p�                                    Bx�e�  �          A$Q�@�\@�z���33�\)B
p�@�\@��o\)����B(��                                    Bx�e��  �          A%p�@�z�@�\)������\B��@�z�@�  �g����\B)33                                    Bx�e�`  �          A%@�@�=q������\)B\)@�@��H����G�B"�                                    Bx�e�  �          A%�@�ff@��������B��@�ff@���33����B3�                                    Bx�e̬  T          A#�
@�(�@����������B{@�(�@�Q������ŅB1��                                    Bx�e�R  �          A%�@��H@��R��Q�����B(�@��H@����W���z�B$Q�                                    Bx�e��  �          A$��@��
@��������
=A뙚@��
@��H��(���\)B�                                    Bx�e��  �          A'
=@��@z=q�����
=Aڏ\@��@��������ffB(�                                    Bx�fD  �          A'\)@�\@`  ��G���A�Q�@�\@��
�����ڏ\B�                                    Bx�f�  �          A(��@�ff@k���  �=qA�=q@�ff@�33��{��\)B�H                                    Bx�f$�  �          A)@�@��
������A�\@�@��R���\��z�B�H                                    Bx�f36  �          A*ff@�
=@n{��33�p�A��@�
=@�{��Q����HB(�                                    Bx�fA�  �          A+
=@�33@n{��G��AЏ\@�33@����ff��z�B                                    Bx�fP�  �          A,  @���@�{��G��(�A噚@���@���������z�B��                                    Bx�f_(  �          A-p�@�\@��
������A�=q@�\@����=q�ӅB�                                    Bx�fm�  �          A.�R@���@�  ��z���A�G�@���@�(����
��  B�                                    Bx�f|t  �          A0z�A Q�@�
=�����A��A Q�@�\)��\)����B\)                                    Bx�f�  T          A2�\A��@�(�������p�A�A��@�\)�mp���33B��                                    Bx�f��  �          A2�HA��@���(���  A��A��@У��j=q���B��                                    Bx�f�f  �          A4Q�A(�@����������RA�G�A(�@�{�p����  B�                                    Bx�f�  �          A4��A��@�
=�����Q�B �RA��@����_\)��=qB                                      Bx�fŲ  �          A4��Aff@�G���33���
A��
Aff@�(��e���{B{                                    Bx�f�X  �          A5��A��@����
��A��
A��@����h������B��                                    Bx�f��  �          A6�\A
{@�����33��p�A��HA
{@�  �g���=qB                                      Bx�f�  �          A6�RA
=q@�{��p���ffA��A
=q@ָR�W
=���\Bp�                                    Bx�g J  �          A<Q�A
�H@�Q���\)��{BG�A
�H@���N{�}��B�
                                    Bx�g�  �          A>=qA�
@�����Q�����B�A�
@�p��L(��x��B!Q�                                    Bx�g�  |          AAA��@�33���H��Q�A�(�A��@��fff��{B�H                                    Bx�g,<  �          AEp�A�
@�{�޸R���A��A�
@�  ��G���B!��                                    Bx�g:�  �          AE@�\)@N{�)��op�A���@�\)@��
����9ffBN�H                                    Bx�gI�  �          AG�
@#�
?��H�@��z�A�@#�
@�\)�-G��s=qB��H                                    Bx�gX.  �          AH  =u>��G�¯�RB���=u@�ff�9���B���                                    Bx�gf�  �          AJff?���?B�\�H��§��B
��?���@�Q��8��B�33                                    Bx�guz  �          AM�=�Q�>�ff�L��¯�)B�B�=�Q�@���=��)B��
                                    Bx�g�   �          ALQ�#�
=#�
�L(�³��B�  �#�
@�p��?�� B�.                                    Bx�g��  �          AK�
>u���K�¯(�C�:�>u@{��AB�                                    Bx�g�l  �          AK�>��E��K33«�HC�j=>�@i���C
=k�B��{                                    Bx�g�  �          AJff?=p��.{�Iª�HC���?=p�@mp��A�(�B�z�                                    Bx�g��  �          AI?^�R�B�\�H��©u�C���?^�R@hQ��@���\B��                                    Bx�g�^  �          AI��?W
=�B�\�H��©��C���?W
=@hQ��@���{B���                                    Bx�g�  �          AI?k��aG��H��¨ffC�!H?k�@a��A�ffB�                                    Bx�g�  �          AIp�?�{��33�G�
¥p�C��?�{@QG��A33B�p�                                    Bx�g�P  �          AF�H?��ÿ���Ep�¦\C���?���@S33�?
=�\B��                                    Bx�h�  
�          AD��?s33�����C33¦��C���?s33@P  �<����B�u�                                    Bx�h�  �          AC33?aG�����A�¦��C�b�?aG�@Mp��;�
W
B�p�                                    Bx�h%B  �          AC\)?B�\�J=q�B�\©�RC��?B�\@`���:�\�qB��                                    Bx�h3�  �          AA�?333�}p��@��¨� C��
?333@S33�:=q�3B�                                    Bx�hB�  �          AA�?&ff����@��¨aHC�?&ff@P���:ff(�B�\)                                    Bx�hQ4  �          AAG�?!G��aG��@z�©C��?!G�@Z=q�9���B�.                                    Bx�h_�  �          AE��?��
���H�B�\ \C�k�?��
@+��?�#�B��                                    Bx�hn�  �          ADQ�?�G���{�@���)C�R?�G�@ ���?
=z�B}=q                                    Bx�h}&  �          AE��?������A���C���?���@{�@���Bj�H                                    Bx�h��  �          AH(�?�z�����D���C��?�z�@(Q��B�Rp�B��                                     Bx�h�r  �          AI?��
�ff�D��C��?��
@���D  ��BU��                                    Bx�h�  �          AG�
?�z��(��C
=��C���?�z�@��B�R�
BW�R                                    Bx�h��  �          AH  ?�{�(��C33�HC�%?�{@�\�B�H�B[��                                    Bx�h�d  �          AHQ�?�(��ff�D(�k�C���?�(�@=q�C33��Bj�\                                    Bx�h�
  �          AH��?�33���D����C���?�33@=q�C�Q�Bo
=                                    Bx�h�  �          AH��?�p���
�D���fC�l�?�p�@{�C��\B}��                                    Bx�h�V  �          AHQ�?��H��33�E�Q�C�8R?��H@(���B�H8RB��)                                    Bx�i �  �          AK�?\��p��G�aHC��)?\@'��Ep�Bo                                    Bx�i�  �          AJff?�  ��=q�G\)��C�=q?�  @1G��Dz�8RB�33                                    Bx�iH  �          AFff?������D��¥�C��\?��@G��?�B�B�.                                    Bx�i,�  �          AE�>W
=�W
=�Dz�«��C�+�>W
=@h���<(�� B�L�                                    Bx�i;�  �          AF=q>��þ��H�F{®�C�>���@�  �;�.B���                                    Bx�iJ:  T          AG�?   �#�
�G\)¬�{C�  ?   @z=q�=��8RB�ff                                    Bx�iX�  �          AHQ�?��333�G�¬
=C�O\?�@w��>=q�B�                                    Bx�ig�  �          AH(�?\)�Q��G�ª�C��?\)@p���>�R��B�(�                                    Bx�iv,  �          AI>Ǯ�����H��©��C��{>Ǯ@c�
�Ap�  B���                                    Bx�i��  �          AK33>���Q��J=q¨� C��>�@_\)�C\)�
B��)                                    Bx�i�x  �          AJ=q?
=�����H��¦W
C��R?
=@Q��C
=k�B��                                    Bx�i�  �          AJ{?(����
�HQ�¥\C�� ?(�@H���C\)��B�{                                    Bx�i��  �          AK
=?@  ���H��£k�C��?@  @A��D���=B�=q                                    Bx�i�j  �          AN{?zῬ���L��¦�C���?z�@Z�H�FffB��f                                    Bx�i�  �          AN�R?
=q����M�¥� C�� ?
=q@P���G�
\)B�
=                                    Bx�iܶ  �          AQ�?z��=q�N�H£{C���?z�@A��K33�B�p�                                    Bx�i�\  �          AR=q?���   �O�¡�3C�` ?��@9���L����B��                                    Bx�i�  T          AR�H?�R��R�O�ǮC��H?�R@,���N=qǮB�#�                                    Bx�j�  
�          AR�\?xQ�����Nff�C�?xQ�@{�N=qB�B�                                    Bx�jN  �          ARff?��R�:=q�L(�.C���?��R@   �O
=ffBh�
                                    Bx�j%�  �          AR{?�p��W
=�J=qk�C��?�p�?���O�¢�3BM                                    Bx�j4�  �          AS
=?����e�I���C�Ff?���?�=q�P��¤\)B?{                                    Bx�jC@  �          AU?p���\)�Q��
=C�K�?p��@!��Qp��B��                                     Bx�jQ�  �          AYp�?&ff�2�\�T���3C���?&ff@�V{u�B�                                      Bx�j`�  �          AX��?��}p��LQ�Q�C�H�?�?���UG�¢�A�z�                                    Bx�jo2  |          AU��@0  ��  �B�\ffC�
=@0  =�G��P��.@p�                                    Bx�j}�  �          A`z�?�G��A��Z=q��C�E?�G�@���\���qBs�
                                    Bx�j�~  �          AdQ�?����@���^=qC��)?���@Q��`(�u�B}�                                    Bx�j�$  �          Af�H?�p��C33�`Q���C��?�p�@���bffu�BiQ�                                    Bx�j��  �          Ag\)?�33�	���c��=C�y�?�33@S�
�`  33B�G�                                    Bx�j�p  �          Ag�
@w���
=�=G��n�C�f@w��aG��QW
C���                                    Bx�j�  �          Ap��A&ff�+������{C�\A&ff��
��Q���  C�1�                                    Bx�jռ  �          Al��A��-����  ���C�J=A���
��ff����C�1�                                    Bx�j�b  �          Ag
=Aff�*�R������z�C��Aff���������C���                                    Bx�j�  �          Ac33A��,z�!G��#�
C�C�A��  ��z����\C���                                    Bx�k�  �          A`��A�R�)���ff��G�C�y�A�R��������HC�K�                                    Bx�kT  �          A\��A  �&{���\����C�t{A  �=q���H��
=C�E                                    Bx�k�  �          AY�A=q�!�33�
�\C��=A=q�	p����R��ffC���                                    Bx�k-�  �          AV�\A
=q� (���R��C�ffA
=q��H���H�Σ�C��)                                    Bx�k<F  �          AR�HAff��\�\)�z�C�%Aff�G���=q�љ�C��                                     Bx�kJ�  �          AP(�A  ����{��C��A  ����Q���ffC���                                    Bx�kY�  �          ALQ�@����
�����C��\@���\)������{C�q                                    Bx�kh8  �          AI�@�\)�{���-�C�aH@�\)� (���33��C���                                    Bx�kv�  �          AG33@��
���p��6ffC��{@��
� ����
=��G�C�8R                                    Bx�k��  �          AD��@�(�����33�C��@�(��z����
��C�`                                     Bx�k�*  �          AA��@�33�녿�\)��\)C�b�@�33�	������=qC�B�                                    Bx�k��  T          A@Q�@ٙ��������(�C�N@ٙ��
�H������{C��                                    Bx�k�v  �          A<��@�=q�33�W
=���C�5�@�=q�{��=q��  C���                                    Bx�k�  �          A:=q@����������
C�}q@���	p���p���p�C�&f                                    Bx�k��  �          A7�@�\)�p��}p���
=C���@�\)�����H��ffC�(�                                    Bx�k�h  �          A4��@����p��333�c�
C���@����	p���=q��{C�=q                                    Bx�k�  �          A1@�Q��녿333�eC��@�Q��	����\���C�J=                                    Bx�k��  �          A/�
@��\�G��O\)��\)C���@��\�z���p����
C�                                      Bx�l	Z  �          A-G�@��
�녿Y����Q�C��@��
����(���
=C�h�                                    Bx�l   �          A+33@�
=��׿\(���33C���@�
=�  ���
���RC�+�                                    Bx�l&�  �          A*=q@�z��z�fff���HC�|)@�z���������\)C�H                                    Bx�l5L  �          A"�\@�z��ff�8Q�����C��3@�z���R��{��z�C�#�                                    Bx�lC�  �          A\)@�ff�Q�>���?ٙ�C�|)@�ff�
=�Mp���Q�C�:�                                    Bx�lR�  �          A�@���
=���8Q�C�u�@����
�^�R��p�C�b�                                    Bx�la>  �          AQ�@��H�{��\)���C�Y�@��H��\�`  ���C�J=                                    Bx�lo�  �          Aff@�p��
�H���H�<(�C��{@�p�����s�
��ffC��                                    Bx�l~�  �          A��@~�R�
�H<#�
=��
C�S3@~�R�   �Vff��=qC�9�                                    Bx�l�0  �          A�
@�
=��
����� ��C�8R@�
=��ff�fff��Q�C�^�                                    Bx�l��  �          A{@�{�{���ÿ��RC�L�@�{���H�c33��{C�t{                                    Bx�l�|  �          A  @����33��(��)��C��@�����z��e���{C�f                                    Bx�l�"  �          AQ�@�
=�p���R�vffC�w
@�
=��R�n{��\)C��f                                    Bx�l��  �          A\)@�p�� �þ��7�C�c�@�p����c�
��{C��
                                    Bx�l�n  �          A��@���� �׾W
=����C��f@�����=q�U����RC��                                    Bx�l�  �          A��@�  ��\)�fffC�#�@�  ����R�\����C�0�                                    Bx�l�  T          A�@|��� �ý��L��C��@|�����
�P����ffC��                                    Bx�m`  �          Aff@�������  �˅C���@����z��S33��(�C�q                                    Bx�m  �          A@�G���(��L�;��
C��
@�G���\)�HQ���p�C��)                                    Bx�m�  �          A��@��R��p����
����C�˅@��R��=q�:�H��  C���                                    Bx�m.R  �          @��
@�33��G���\�p��C���@�33��{�(����C�j=                                    Bx�m<�  �          @�@���"�\�#�
���C�)@�����˅�B�HC�Ǯ                                    Bx�mK�  �          AQ�@�����ý�Q�(�C��@�����5��(�C�33                                    Bx�mZD  �          @�@������;������C�e@�����  �8����p�C��f                                    Bx�mh�  �          @��
@���ҏ\�L�;�{C���@�������(Q���Q�C���                                    Bx�mw�  �          @�  @q���  <�>B�\C���@q��ƸR�(Q���33C��f                                    Bx�m�6  �          @��\@h����p�>�@aG�C��q@h���љ�������C��f                                    Bx�m��  �          @�G�@b�\����?333@�z�C���@b�\���
���w�C�{                                    Bx�m��  �          @��@x����z�?E�@�z�C��=@x����p����_
=C��                                    Bx�m�(  �          @��@w
=�θR?.{@��C�k�@w
=�ƸR��
=�lz�C��\                                    Bx�m��  �          @��@`  ��\)?��@�z�C�33@`  ��� ���|��C�                                    Bx�m�t  �          @�=q@aG���(�>B�\?�G�C��3@aG���ff�  ����C��)                                    Bx�m�  �          @��@Dz�����>u@   C�|)@Dz���(��
�H����C�J=                                    Bx�m��  �          @�=q@aG���
=���
�333C���@aG���
=���G�C���                                    Bx�m�f  �          @Ӆ@�p����\�\�UC�<)@�p������G����C��{                                    Bx�n
  �          @У�@�=q���ͽ��
�333C�f@�=q��Q������G�C�\)                                    Bx�n�  �          @Ϯ@����=q�8Q���C���@���x�ÿ�����C��                                    Bx�n'X  �          @�R@g��Ϯ>�@aG�C��3@g���z��
=q����C�C�                                    Bx�n5�  �          @���@W���Q�?u@�ffC��=@W���33���H�V{C��{                                    Bx�nD�  �          @�R@2�\���?Y��@ڏ\C���@2�\�˅��=q�l(�C���                                    Bx�nSJ  �          @�G�@1G���(�?:�H@�  C��q@1G����Ϳ�\)�x(�C�%                                    Bx�na�  
r          @��\@�p��
=q?�G�A/\)C�H@�p��fff?5@�p�C��                                    Bx�np�  
�          @y��@8�ÿ��?�=qA�  C�h�@8���
=?0��A'\)C�aH                                    Bx�n<  
Z          @]p�?�G���
?��
B�C��?�G��5�?5AH��C�
                                    Bx�n��  
�          ?:�H>��H>\)>�{Bp�A�  >��H�#�
>�p�B\)C���                                    Bx�n��  
�          @#33?޸R?�G��8Q���{B5{?޸R?�
=?
=qAH  B0�                                    Bx�n�.  �          ?�ff?�G�?�=q<#�
>�ffBS�\?�G�?�(�?��A�ffBIQ�                                    Bx�n��  T          @333@ ��?�(�����L��B!�H@ ��?�>\)@9��B(�                                    Bx�n�z  T          @G�?�33?��ÿ
=�qA��?�33?�(���G��1G�B�
                                    Bx�n�   �          ?��?k�?+�����{Bff?k�?L�;#�
��(�B$=q                                    Bx�n��  "          ?\?O\)?Tz��ff����B6��?O\)?p�׽��
�i��BD�                                    Bx�n�l  
�          ?��׽��
����?.{A��RC����
���>.{@�RC�J=                                    Bx�o  T          @��k����?�G�A��C����k��Q�>.{@��HC��\                                    Bx�o�  
B          @{�8Q��	��?n{A��HC��=�8Q���#�
��=qC��)                                    Bx�o ^            @���?.{���=u?G�C��3?.{�r�\��������C�R                                    Bx�o/  T          @�
=���H��33?E�A�C�˅���H��G�����YC��H                                    Bx�o=�  
�          @��׿!G����?!G�@�p�C�#׿!G����׿�ff�}�C�                                    Bx�oLP  �          @��?�=q�.{��33�8�C�n?�=q>�녿����0G�A��                                    Bx�oZ�  T          @*=q?�����z��VffC�k�?�>������]�AQG�                                    Bx�oi�  
�          @��?���@P������\)B���?���@x�ÿ@  �)�B�ff                                    Bx�oxB  "          @��?��\@w
=�.�R�	\)B�W
?��\@�z�p���2�HB��                                    Bx�o��  
�          @h��?��H?����-p��D
=BO
=?��H@2�\��p����By�\                                    Bx�o��  "          @}p�@��@*�H���
���HB=�\@��@C�
���
����BM                                      Bx�o�4  "          @J=q?�p�@#33�aG�����B_(�?�p�@�H?Tz�AxQ�BY��                                    Bx�o��  "          @Fff@�?�=q?k�A��B�H@�?���?��RA��A¸R                                    Bx�o��  
�          @!G�?���?0��?��A�A��?���>W
=?�p�B�H@Ϯ                                    Bx�o�&  
�          @7�?�ff?�p�?=p�Aw\)B>�
?�ff?�G�?�p�B �RB�                                    Bx�o��  
�          @a�?�{��(�@
=qB(G�C�O\?�{�+�?�
=A��C�xR                                    Bx�o�r  �          @Z=q?�G��W
=@1G�Bu
=C�(�?�G����@(�B/�\C��)                                    Bx�o�  �          @�p�>.{@C�
?��A�=qB���>.{@
=@\)B-�RB���                                    Bx�p
�  �          @��H���@��ÿ^�R�أ�B�ff���@�  @
=qA��B��3                                    Bx�pd  "          @��;��@�\)���
�?�B�ff���@�\)?��A?�
B�ff                                    Bx�p(
  �          @�(��aG�@���
=�RffB�p��aG�@�  ?��A-�B�k�                                    Bx�p6�  �          @�녾�  @����{�+�B���  @��H?�AT  B�\                                    Bx�pEV  
�          @�\)>aG�@�?0��@���B�G�>aG�@���@\��A��B��q                                    Bx�pS�  �          @�{>.{@���?5@�(�B��>.{@�Q�@]p�A��B��                                    Bx�pb�  "          @�p�>�\)@��
?�@���B�aH>�\)@�=q@QG�A�z�B�                                    Bx�pqH  �          @�33>���@��H���
�0��B��f>���@���@-p�A��RB�p�                                    Bx�p�  
�          @�33��@�녿&ff��{B�z��@�\)@�A���B��                                     Bx�p��  	�          @��ý#�
@�
=�k�����B��3�#�
@׮@   A�Q�B��q                                    Bx�p�:  
Z          @�G�=#�
@�Q�+����RB�L�=#�
@�p�@  A���B�B�                                    Bx�p��  T          @�
=��\)@��5���B�.��\)@��H@�A�p�B�=q                                    Bx�p��  T          @��
>��R@��H������B�.>��R@�p�@ ��A�p�B��f                                    Bx�p�,  
(          @�{>���@���
=q��B���>���@�\)@!�A�33B�p�                                    Bx�p��  
�          @�p�>#�
@�(��E���
=B�p�>#�
@ᙚ@�
A�33B�W
                                    Bx�p�x  
�          @�\>8Q�@��ÿJ=q�ƸRB�(�>8Q�@�
=@  A�\)B�
=                                    Bx�p�  	�          @�(�>�\)@�33�����  B���>�\)@�@ ��A�
=B�ff                                    Bx�q�  �          @�  >�G�@�\)<�>uB�L�>�G�@��@EA��HB���                                    Bx�qj  T          @���>��
@�׽L�;��B�.>��
@�(�@A�A��B��q                                    Bx�q!  �          @���>W
=@��>8Q�?�\)B���>W
=@�G�@O\)A�Q�B�u�                                    Bx�q/�  �          @���>B�\@��>��?��HB��>B�\@�  @Tz�A��B���                                    Bx�q>\  
�          @��>B�\@��=�?h��B��>B�\@��@L(�A���B���                                    Bx�qM  
�          @�  >u@��#�
��  B�ff>u@���@:�HA��B�{                                    Bx�q[�  
�          @�Q�>��@��k����
B���>��@�@7
=A�\)B���                                    Bx�qjN  
�          @�Q�>\@���\)�
=B�B�>\@�{@4z�A���B���                                    Bx�qx�  �          @�>��R@�\)����� ��B�33>��R@�ff@1G�A���B��
                                    Bx�q��  �          @�Q�>���@����
�(�B���>���@޸R@1�A�(�B��                                     Bx�q�@  "          @�Q�>�(�@������#33B�k�>�(�@�
=@1G�A�G�B��                                    Bx�q��  �          @�Q�?�@�\)��=q�B�W
?�@�{@5�A���B���                                    Bx�q��  �          @��?&ff@�>.{?�G�B��?&ff@�Q�@N�RA�p�B�
=                                    Bx�q�2  "          @�Q�?+�@�
=>��@g
=B��R?+�@Ӆ@`  A�(�B�ff                                    Bx�q��  �          @�Q�>W
=@���ff�^{B��
>W
=@�Q�@*�HA�ffB���                                    Bx�q�~  "          @�G�>��@�  �G����RB���>��@���@�A���B��=                                    Bx�q�$  �          @���>�
=@�  �\�9��B���>�
=@߮@/\)A��HB�33                                    Bx�q��  T          @�>�p�@�\)��\)��B�aH>�p�@�33@@��A���B��H                                    Bx�rp  �          @�
=>\@�ff=�\)?�B��>\@�Q�@HQ�A�z�B��                                     Bx�r  T          @�>�Q�@�ff=�Q�?5B�z�>�Q�@�  @I��A��B��                                    Bx�r(�  
Z          @�>�
=@�ff�
=����B��\>�
=@�G�@!�A�{B�.                                    Bx�r7b  
(          @�=q?
=q@�  �L����G�B��=?
=q@�{@��A�B�(�                                    Bx�rF  �          @�Q�?�\@�ff�E����
B��?�\@�(�@��A�33B��\                                    Bx�rT�  �          @�p�?�@��H�G���Q�B�z�?�@���@p�A�(�B�{                                    Bx�rcT  �          @�?�@��H�k����B�G�?�@Ӆ?�p�A���B���                                    Bx�rq�  "          @ڏ\?(�@�
=��ff��B���?(�@љ�?�Av{B�aH                                    Bx�r��  
Z          @�  >��@��Y����{B�.>��@��@A��HB��)                                    Bx�r�F  �          @�z�>�{@�33��R��{B�G�>�{@�  @��A��B���                                    Bx�r��  �          @�ff>�G�@�(��Y����Q�B���>�G�@��
@z�A�z�B�Q�                                    Bx�r��  �          @�\)?�\@���Q���
=B���?�\@�z�@
=A��RB�G�                                    Bx�r�8  T          @�ff?   @��
�p�����B��3?   @�z�?�p�A��\B�ff                                    Bx�r��  �          @��>B�\@߮�&ff����B���>B�\@�(�@33A��RB���                                    Bx�r؄  T          @߮>�p�@޸R��
=�\(�B��H>�p�@�Q�@\)A�G�B�u�                                    Bx�r�*  �          @�{?�\@������ffB��=?�\@��H@/\)A�(�B���                                    Bx�r��  
Z          @��>�(�@��?G�@�  B��R>�(�@�(�@aG�A���B��{                                    Bx�sv  �          @ڏ\>�{@�  ?   @���B�L�>�{@�{@P  A�
=B��=                                    Bx�s  �          @�(�>�z�@ڏ\?\)@�{B��>�z�@��@U�A�z�B�k�                                    Bx�s!�  f          @��
?c�
@׮����]p�B���?c�
@��@=qA��
B��                                    Bx�s0h  �          @�z�?��
@������Q�B�8R?��
@���?�(�Ag\)B���                                    Bx�s?  �          @���?��R@Ӆ���
�Lz�B�?��R@���?��A3�B��H                                    Bx�sM�  �          @�=q?�ff@�zῃ�
�33B�?�ff@�\)?�Av�\B��\                                    Bx�s\Z  �          @ڏ\?W
=@׮�(�����B��R?W
=@��@�A�G�B���                                    Bx�sk   �          @�=q?E�@�  �\)��\)B��?E�@�(�@�A�=qB�(�                                    Bx�sy�  �          @ٙ�?@  @׮���H���B�Q�?@  @��H@A��HB��=                                    Bx�s�L  �          @�=q?!G�@أ׿z���33B�ff?!G�@���@G�A��B���                                    Bx�s��  �          @ڏ\?L��@�\)�E���ffB�k�?L��@�ff@A�ffB��
                                    Bx�s��  �          @���?W
=@�p��G��ӅB���?W
=@���@33A�
=B�                                    Bx�s�>  �          @�G�>L��@�Q�#�
���
B��\>L��@��@2�\A£�B�8R                                    Bx�s��  �          @��H=�Q�@�p�<�>uB�� =�Q�@��@1�A�\)B�W
                                    Bx�sъ  �          @��
?��@�z῕��B�(�?��@���?�
=Ab{B���                                    Bx�s�0  �          @�z�?�G�@�����R�%G�B���?�G�@�=q?У�AZ�HB�\)                                    Bx�s��  �          @�ff?��H@�  �����=qB��3?��H@Ӆ?�G�Aj�\B�L�                                    Bx�s�|  �          @�?�Q�@׮��ff�(�B���?�Q�@�=q?�=qAt��B�z�                                    Bx�t"  �          @�?��@�{���
�*{B�.?��@��
?���AU�B��                                    Bx�t�  �          @޸R?�ff@�
=���R�$��B��?�ff@�z�?�33A[33B���                                    Bx�t)n  �          @�
=?��R@�p���=q�/�B���?��R@��
?�ffAN=qB���                                    Bx�t8  �          @�{?��
@�\)��Q��aB�\)?��
@�33?��A�
B��H                                    Bx�tF�  �          @�\)?��
@љ���{�U�B���?��
@�(�?�  A$��B���                                    Bx�tU`  �          @�z�?�@�녿����0��B�u�?�@�Q�?�G�AJ�\B�G�                                    Bx�td  �          @׮?��\@Ӆ�5��33B�k�?��\@��@�A�Q�B���                                    Bx�tr�  �          @���<�@�33?�@�{B�u�<�@���@P  A���B�ff                                    Bx�t�R  �          @�{=�G�@�p�>���@5B�
==�G�@�{@C�
A�(�B���                                    Bx�t��  �          @�(�>�=q@Ӆ>��@G�B�G�>�=q@��@>{A�\)B��R                                    Bx�t��  �          @�p�>\@���=u?�B�ff>\@���@3�
A�G�B��R                                    Bx�t�D  �          @�p�>aG�@��
?&ff@���B�B�>aG�@�Q�@U�A�ffB��3                                    Bx�t��  �          @�(�>#�
@��H?5@��B�B�>#�
@�ff@W�A���B��
                                    Bx�tʐ  �          @�p���33@�z�>��@�Q�B����33@�33@J�HA��B��                                    Bx�t�6  �          @�p�����@ҏ\?p��AB��;���@��@e�B��B���                                    Bx�t��  �          @����@Ϯ?��A9�B�aH��@��@y��B�
B�#�                                    Bx�t��  �          @�  ��R@��@A��
B����R@��@��B5�Býq                                    Bx�u(  �          @���=u@��H?5@��B���=u@�ff@W�A�\B���                                    Bx�u�  �          @�{?!G�@�33�\)����B�\?!G�@�=q@&ffA��RB��                                    Bx�u"t  �          @�
=?�(�@�
=��  ���B�p�?�(�@��?�G�At  B���                                    Bx�u1  �          @���@z�@ƸR�����  B��@z�@�33?˅A^�\B�z�                                    Bx�u?�  �          @���@,��@������:�RB��@,��@�{?�(�A)p�B��                                    Bx�uNf  �          @�{?��H@�녿���p�B�aH?��H@�
=?�G�AZffB��                                    Bx�u]  �          @�p�?�p�@θR�Tz���{B�aH?�p�@�\)?�z�A�(�B��{                                   Bx�uk�  �          @�
=?�  @�33��p��)��B�33?�  @�G�?�  AN�HB��                                   Bx�uzX  �          @�{?�{@θR�z���ffB�\?�{@��
@��A���B��{                                    Bx�u��  �          @���?�\)@�
=�#�
����B�{?�\)@��@A�Q�B��f                                    Bx�u��  �          @�
=?�=q@�녾��z�HB�
=?�=q@�p�@�\A���B���                                    Bx�u�J  �          @ٙ�?���@���B�\�˅B��{?���@�z�@%�A�
=B��                                    Bx�u��  �          @ڏ\?��@�p�>���@UB�ff?��@��@G
=A��B�z�                                    Bx�uÖ  �          @ָR?���@��H�W
=��=qB�� ?���@��H@!G�A�33B��                                    Bx�u�<  �          @أ�?��@�=q��p��'�B�\?��@Ϯ?˅AYG�B��)                                    Bx�u��  �          @ָR?Q�@љ��p����B�Ǯ?Q�@�33?���A��RB�W
                                    Bx�u�  �          @�z�?��@��ÿh����33B��q?��@љ�?�(�A�\)B�k�                                    Bx�u�.  �          @ۅ?=p�@ٙ�<��
>8Q�B�� ?=p�@�{@4z�A��HB�G�                                    Bx�v�  �          @�z�?O\)@ҏ\�L�Ϳ�Q�B�\?O\)@\@!�A���B��f                                    Bx�vz  �          @���?.{@љ��W
=��G�B��?.{@�=q?�
=A�G�B���                                    Bx�v*   �          @��?n{@Ϯ��Q�Q�B��q?n{@�ff@%A��B�L�                                    Bx�v8�  �          @У�?aG�@θR�������B��?aG�@�ff@!G�A�  B�8R                                    Bx�vGl  �          @�33?��H@Ϯ�G�����B�u�?��H@Ǯ?��HA��RB�Q�                                    Bx�vV  �          @�@ff@��H���
�K�B�W
@ff@���?�(�A!B��R                                    Bx�vd�  	�          @׮?���@����33��G�B��H?���@���?L��@�=qB��                                    Bx�vs^  �          @ڏ\@{@�����
��=qB�z�@{@�>�p�@HQ�B��                                    Bx�v�  T          @��?�  @�z�&ff��ffB��=?�  @�=q@��A�B�L�                                    Bx�v��  �          @���?��@��
>�
=@\��B�{?��@��H@Mp�AمB�8R                                    Bx�v�P  T          @�  ?c�
@�p�>�33@5�B�Q�?c�
@�p�@J=qA�
=B��\                                    Bx�v��  �          @�(�>��
@׮>��R@(��B��\>��
@���@C33A��HB��                                    Bx�v��  "          @��?��@�=q�ff��Q�B���?��@Ϯ>�33@=p�B�                                    Bx�v�B  
�          @��?�=q@�(��.{���B�G�?�=q@ʏ\@ffA�{B�33                                    Bx�v��  �          @�\)?�\)@�G��Q�����B�G�?�\)@ə�?�
=A�p�B�k�                                    Bx�v�  
�          @�Q�?��@�Q�Tz����B�?��@���?�A��B��q                                    Bx�v�4  
�          @ڏ\?\@�(��
=q��Q�B�.?\@ȣ�@�RA��\B��q                                    Bx�w�  
�          @�=q?�p�@�ff��Q��@  B�.?�p�@�  @�HA�Q�B�                                    Bx�w�  "          @أ�?�ff@�33�0�����
B�?�ff@ə�@z�A���B��q                                    Bx�w#&  �          @�  ?��R@�=q���R�(Q�B�z�?��R@��
@=qA�p�B��                                    Bx�w1�  �          @�Q�?��@�G����H��B��f?��@��@�RA�  B�G�                                    Bx�w@r  "          @�G�?��H@�
=��z���RB�W
?��H@���@Q�A��HB�                                      Bx�wO  �          @�{?�@����\���
B��?�@���@
=qA���B�Ǯ                                    Bx�w]�  
�          @��?��H@��Ϳ.{����B�#�?��H@��
?��RA�\)B��
                                    Bx�wld  �          @�ff?��H@��þ�33�B�\B���?��H@�33@A��B��                                    Bx�w{
  T          @�  ?�(�@ҏ\��  �Q�B��
?�(�@�33@p�A��B���                                    Bx�w��  �          @�(�?�(�@�Q�aG����B��
?�(�@���@p�A�ffB�33                                    Bx�w�V  �          @�z�?�{@�\)=�?�=qB�W
?�{@��@0��A��B��                                    Bx�w��  �          @�\)?�ff@ҏ\>�p�@J�HB���?�ff@��@A�A�{B���                                    Bx�w��  T          @�p�?��@أ׾�\)�33B���?��@ə�@ ��A�z�B�                                      Bx�w�H  �          @ٙ�?���@��H�E���=qB��H?���@ʏ\?�p�A�  B���                                    Bx�w��  �          @�z�?���@�Q�O\)���B��?���@�G�?�A�G�B���                                    Bx�w�  
�          @�=q?�{@�{@#33A���B���?�{@�z�@�  B9z�B��{                                    Bx�w�:  �          @أ�?��@��H@�A�Q�B�L�?��@�p�@���B'�RB�\                                    Bx�w��  �          @��
?��
@У�?�p�Ai�B��q?��
@�  @�\)B33B��=                                    Bx�x�  
�          @��H?�ff@Ӆ?��Ap�B�Ǯ?�ff@�=q@n�RB\)B���                                    Bx�x,  "          @ۅ?�
=@�z�?�  AffB�?�
=@��@g�A�Q�B��3                                    Bx�x*�  T          @���?�Q�@ҏ\?z�@���B�W
?�Q�@���@L��A�=qB�
=                                    Bx�x9x  "          @׮?�\)@�G�>�  @��B�B�?�\)@�(�@8��A��HB�=q                                    Bx�xH  �          @��?���@ҏ\?B�\@�ffB�8R?���@��R@W�A�Q�B��\                                    Bx�xV�  �          @��
?���@У�?���AC
=B��?���@��
@~{Bp�B�ff                                    Bx�xej  
�          @���?�z�@�z�?O\)@�\)B��f?�z�@��@[�A���B��3                                    Bx�xt  �          @��H?��
@��?5@�
=B��\?��
@��R@S�
A�B�G�                                    Bx�x��  "          @ָR?�33@���?(�@�B��H?�33@�33@I��A�  B��                                     Bx�x�\  
�          @��?���@˅>�\)@�B��f?���@�ff@5�AɅB�8R                                    Bx�x�  �          @�ff@�@��
=�?�  B��q@�@���@+�A���B�L�                                    Bx�x��  �          @�\)@z�@�녽����
B��{@z�@�=q@(�A�ffB�z�                                    Bx�x�N  �          @�{?�G�@�>�(�@n{B�W
?�G�@��R@@  A�33B���                                    Bx�x��  	�          @��?��@��?uA z�B��?��@�ff@dz�A�p�B��R                                    Bx�xښ  �          @��H?���@Ϯ?��HAF�\B��)?���@�33@}p�B=qB��R                                    