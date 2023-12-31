CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230411000000_e20230411235959_p20230412021719_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-12T02:17:19.365Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-11T00:00:00.000Z   time_coverage_end         2023-04-11T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxv�ˀ  �          A(���
=@���@^{A���C�H��
=@33@�B=qC p�                                    Bxv��&  �          Ap���Q�@�33@`  A�z�CL���Q�@
=@�  BC 
                                    Bxv���  �          A��
=@���@dz�A�{C�H��
=@
=@��\B{C��                                    Bxv��r  T          A��θR@���@Mp�A�Ck��θR@1G�@���B33C�\                                    Bxv�  �          A=q��
=@�ff@J�HA�G�C:���
=@5�@�z�Bz�Cff                                    Bxv��  �          A�R��(�@��@6ffA�33C����(�@A�@�z�BCu�                                    Bxv�#d  
�          A���{@���@)��A��C���{@<��@��BG�C�3                                    Bxv�2
  
�          A���=q@�ff@1G�A�ffC��=q@4z�@�p�Bp�C=q                                    Bxv�@�  �          Az����H@��@6ffA�
=Cٚ���H@*=q@��B�HCff                                    Bxv�OV  
�          A
=����@��R@L(�A��C�R����@��@���B�RC �\                                    Bxv�]�  �          A����@���@G
=A�ffCQ����@ ��@��
Bp�C 
=                                    Bxv�l�  �          A�H�أ�@��\@N{A�p�C���أ�@\)@�  B  C��                                    Bxv�{H  T          A��z�@��@K�A�Q�C����z�@%@��BffC�R                                    Bxvډ�  �          A��G�@�\)@0  A��\C�R��G�@7�@��B�RC{                                    Bxvژ�  "          A����{@��\@%�A���C.��{@5�@�{B 
=C�
                                    Bxvڧ:  T          A����Q�@��@'�A��RC����Q�@<(�@���Bp�Cz�                                    Bxvڵ�  "          A���  @�ff@0��A��
C���  @5@���B�C33                                    Bxv�Ć  T          A����z�@�Q�@.�RA���CY���z�@,(�@���B��C��                                    Bxv��,  �          A  ����@��@+�A�z�C������@(��@�Bz�C�                                    Bxv���  T          A�����
@�ff@3�
A�G�C�����
@&ff@�=qB
=C@                                     Bxv��x  
�          A�߮@��@7
=A���CW
�߮@"�\@��HB33C 
=                                    Bxv��  �          A�
��ff@��@&ffA�z�C�R��ff@AG�@�=qB�C��                                    Bxv��  "          A  ��{@�  ?�p�A6�HC5���{@Z�H@�G�Aܣ�C�                                    Bxv�j  �          A���Q�@�z�?�G�A9�C���Q�@S�
@�  Aڣ�C��                                    Bxv�+  
�          A
�H����@���?�p�A7�C�3����@O\)@{�A�p�C=q                                    Bxv�9�  
�          A���=q@��?�(�A6{C����=q@P��@{�A�Q�CE                                    Bxv�H\  
�          A(����@�z�?ٙ�A3
=CE���@U@|��A���C�f                                    Bxv�W  
�          A����H@�p�?�A=C:����H@S�
@��Aۙ�C�3                                    Bxv�e�  
          A���
=@�?�G�A:ffC�R��
=@Vff@���A�  CW
                                    Bxv�tN  �          A
=��(�@��?�\A<��C\��(�@X��@�=qA�\)C                                    Bxvۂ�  "          A
=q�أ�@�p�?�(�AQ�C�f�أ�@l��@x��A�  CQ�                                    Bxvۑ�  �          A
�R�ҏ\@��R?�Q�A��CT{�ҏ\@~{@�Q�A܏\C�f                                    Bxv۠@  �          A(��Ϯ@��@#�
A�  C�=�Ϯ@Fff@���B  Cz�                                    Bxvۮ�  
�          A	���33@�  @A�A�Q�C�\��33@1G�@�z�BffCs3                                    Bxv۽�  T          A	�����@��@�AfffC������@\(�@��A�C�q                                    Bxv��2  �          A����Q�@�?���@�
=C&f��Q�@�z�@k�A�{C�\                                    Bxv���  �          A	���z�@�  ?}p�@�p�C����z�@���@`��A��HCǮ                                    Bxv��~  �          A����z�@���?�=q@�\)Cc���z�@���@g
=A�C�                                    Bxv��$  �          A(���  @�Q�?���A)C�q��  @o\)@���A�\)C
=                                    Bxv��  "          A	��z�@��\?���A((�Cz���z�@�\)@���A�  Ch�                                    Bxv�p  �          A	G���33@��?��A0(�CY���33@�{@��A�  C�                                    Bxv�$  "          A	p����@��R?�  A<��C	.���@�G�@���A��RC�q                                    Bxv�2�  
�          Az����H@�=q?�(�A\)C
�����H@�{@w
=Aי�C�=                                    Bxv�Ab  
�          A  �θR@��?�@�(�C{�θR@��\@o\)A�z�C�q                                    Bxv�P  
(          Az���p�@�{?�ffA'�C	O\��p�@�(�@��RA�\C33                                    Bxv�^�  "          A  ��  @���?ٙ�A8Q�C���  @�(�@�(�A�{Cu�                                    Bxv�mT  "          A�R���H@�z�@��Ar=qCz����H@~�R@��
Bp�C��                                    Bxv�{�  
�          A�
����@��?�=qA,Q�CQ�����@r�\@�G�A�(�C^�                                    Bxv܊�  
�          A��(�@�p�?�A9�C
�=��(�@u@�p�A��C�R                                    BxvܙF  "          A{��
=@�(�?���A/�
C!H��
=@vff@��HA�CB�                                    Bxvܧ�  �          A����z�@��?�=qAE�C	����z�@{�@���A��HC\)                                    Bxvܶ�  "          A����\)@�z�@`  A��CT{��\)@+�@��B"=qC�
                                    Bxv��8  T          A  ���@��\@c�
A�  C
�
���@&ff@�(�B%�HCٚ                                    Bxv���  �          A����\@��@_\)A�ffC����\@(Q�@��B#��C��                                    Bxv��  T          Az��љ�@�z�@{Ar�HC@ �љ�@G�@��
A��\C�=                                    Bxv��*  �          A
�R��G�@��R?n{@���C�
��G�@u�@R�\A��\Cz�                                    Bxv���  �          A
{����@�ff?�@���C}q����@z�H@g�AƏ\C��                                    Bxv�v  T          A	p���33@��?�=q@�
=C����33@u@^{A��RC�3                                    Bxv�  �          A���ָR@��
?��
A	��C��ָR@r�\@j=qA�p�C�=                                    Bxv�+�  �          A���{@�=q>�  ?�Q�C����{@u�@   A�  C\                                    Bxv�:h  �          A����{@�G�@'�A���Cz���{@Q�@��B33C!H                                    Bxv�I  
�          A(���
=@��@\)At��CG���
=@aG�@��
Bz�C�                                     Bxv�W�  �          Az���ff@��R@ffAd��C
�f��ff@k�@���B ��CQ�                                    Bxv�fZ  �          A	G���\)@��@\)Ar�\CxR��\)@Vff@���A�(�C�f                                    Bxv�u   "          A	p����@��R?�33AL(�C�
���@dz�@��A��Cz�                                    Bxv݃�  T          A
�H���@�=q@AG�A�G�C8R���@b�\@���B=qC��                                    BxvݒL  T          A33��Q�@��@s�
A�(�C�f��Q�@S33@�G�B9ffC�H                                    Bxvݠ�  "          A\)����@��@p��A���B�p�����@l(�@�ffB?ffC��                                    Bxvݯ�  �          A33���@�ff@j�HA��
C ����@c�
@ȣ�B8��C�=                                    Bxvݾ>  �          A33���@���@n{Aʏ\Cp����@L(�@ÅB2�C                                      Bxv���  
�          AQ���\)@�(�@Dz�A�{C����\)@W
=@��RB(�C��                                    Bxv�ۊ  
�          A����@���@E�A��C	Y����@QG�@��B��Cff                                    Bxv��0  "          A  ��Q�@��\@>�RA�G�C�
��Q�@J=q@�\)B�HCE                                    Bxv���  
�          A(���z�@��\@3�
A�C	����z�@]p�@��RB�HC�)                                    Bxv�|  "          A����=q@��
@=qA~{C
^���=q@l(�@��
B�\C��                                    Bxv�"  T          A�
��@�p�@(�A��C	p���@n{@�B	ffC�                                    Bxv�$�  T          A���Å@��R@.�RA���C�f�Å@g�@�ffB�C^�                                    Bxv�3n  
�          A�H��33@��@7
=A�\)C(���33@k�@�(�B33C��                                    Bxv�B  �          A�\��@��
@C�
A�G�CB���@fff@��B  C�                                     Bxv�P�  "          AG����@�(�@w�AϮC�����@#�
@���B'z�C�                                    Bxv�_`  �          A(���
=@��@aG�A�C�
��
=@*�H@��B�C�                                    Bxv�n  T          A(����@�z�@G�A�
=C
�=���@J=q@��
BQ�C�)                                    Bxv�|�  T          A����{@���@,��A��\C	�H��{@e@�(�BffC��                                    BxvދR  
�          AG���Q�@���@33Aq�C�f��Q�@��
@�
=B	(�C��                                    Bxvޙ�  T          A
=�Ӆ@��@!G�A�G�C���Ӆ@G
=@�z�BC��                                    Bxvި�  "          A\)���
@�
=@�Ax(�CW
���
@g�@��RB{CaH                                    Bxv޷D  �          A���(�@�  @(�AiG�C33��(�@n{@��A��
C��                                    Bxv���  �          A33��=q@�p�?��
A>=qC����=q@L��@vffA���C�f                                    Bxv�Ԑ  �          A\)��{@��\?�=qAB=qC���{@~{@��\A�z�CY�                                    Bxv��6  �          A
�\��  @�\)?�G�A;�
C	� ��  @�(�@��HA��C��                                    Bxv���  "          A	����33@�{@�
A]C�R��33@}p�@��\B ��C                                      Bxv� �  
�          A(��Å@���?�G�A?
=C	=q�Å@�=q@�G�A�RCT{                                    Bxv�(  "          A(��ə�@��?�AD��CO\�ə�@u@�
=A�ffC��                                    Bxv��  
�          Az����@�\)@
=Aep�C�=���@a�@�z�A�  C#�                                    Bxv�,t  b          AQ����
@�(�?ٙ�A8��C�
���
@w�@��A��HC                                    Bxv�;  �          A����
=@��H?�ffA(  C}q��
=@y��@}p�A��C�f                                    Bxv�I�  �          A
{�У�@��R?�Q�A�C��У�@�=q@z�HA��C�                                    Bxv�Xf  
�          A
{��  @��
?�\)AHQ�Cs3��  @q�@�  A뙚C��                                    Bxv�g  �          A	G���(�@�\)@,(�A�(�C� ��(�@5�@���B=qC�f                                    Bxv�u�  
�          A(����@��@=p�A���C� ���@-p�@�z�BQ�C�                                    Bxv߄X  b          A33����@�{@5A�C������@.{@�Q�B	�C                                      Bxvߒ�  �          A\)�ָR@��
@.�RA�ffC���ָR@   @�  B �\C�
                                    Bxvߡ�  
�          A����(�@��@>{A���C^���(�@�H@�\)B
�C@                                     Bxv߰J  
�          A�\��33@�=q@$z�A��RC:���33@w
=@��Bz�C�                                    Bxv߾�  �          A��g�@��
@�A�\)B�{�g�@�z�@��B%p�B�                                    Bxv�͖  "          @�{�r�\@�{@&ffA�G�B���r�\@�z�@�z�B+�\C�\                                    Bxv��<  �          @�\)��Q�@I��@HQ�A���CG���Q�?�  @���B��C(�R                                    Bxv���  "          @�
=��{@=q@X��A؏\C� ��{>��@�(�B��C/�H                                    Bxv���  
�          @�=q��{@Q�@0  A�ffCQ���{?B�\@c�
A�Q�C,h�                                    Bxv�.  
�          @ҏ\����@�z�@{A�C������@*=q@���B%�C�q                                    Bxv��  T          @�=q?�(�@��R?\Aj{B�\)?�(�@��@|��B$�B�G�                                    Bxv�%z  
�          @�{@7�@�p�=��
?5Bx\)@7�@�
=@*�HA�ffBj�                                    Bxv�4   �          @Ǯ@AG�@�p���Q��UBs�\@AG�@�@�\A�z�Bj{                                    Bxv�B�  
�          @��@<(�@�p�>W
=?�Bz\)@<(�@�(�@:=qA�ffBk��                                    Bxv�Ql  �          @��
@@��@��=L��>�Bw=q@@��@���@.�RA�\)Bi�
                                    Bxv�`  
Z          @У�@?\)@���<�>aG�Bz�@?\)@��@2�\A���Bm�                                    Bxv�n�  
�          @�G�@0��@�33?z�@��
B��=@0��@��@S�
A��HBq��                                    Bxv�}^  
�          @��
@>�R@��>�=q@�B|{@>�R@���@A�AۮBm{                                    Bxv��  T          @�z�@��@�p�?Y��@�z�B�@��@���@l(�B�B���                                    Bxv���  
�          @�ff@�R@���?
=q@�p�B��@�R@���@Z=qA��\B���                                    Bxv�P            @�p�@5�@�ff?��\A�B��@5�@�
=@vffBBqG�                                    Bxv��  0          @߮@P  @�p�@�A�\)Bp�@P  @w
=@���B-�
BG�\                                    Bxv�Ɯ  �          @�
=@U@��
@A��
Bl�@U@u�@�\)B+�
BCff                                    Bxv��B  �          @�\)@1�@��?�Q�A��B���@1�@��@��HB%\)Bf��                                    Bxv���  T          @�\@*�H@���?���AnffB�  @*�H@��@��\B!�RBq
=                                    Bxv��  	�          @�Q�@	��@��?��Ar=qB�B�@	��@�
=@��\B&�HB�8R                                    Bxv�4  �          @�G�@��@�(�@!�A��B��\@��@�ff@�(�B<p�Bv��                                    Bxv��  
Z          @�\@,��@��@   A�G�B��=@,��@~�R@�\)B7��B_p�                                    Bxv��  
�          @�@���@�  ?Tz�@���B>��@���@�\)@Mp�A�B&ff                                    Bxv�-&  
�          @��@qG�@�?�z�A��Bf{@qG�@�@tz�B�RBLp�                                    Bxv�;�  
(          @�Q�@X��@���?�ffA$��Bvz�@X��@�@��B	�
B]��                                    Bxv�Jr  
~          @�@fff@ə�?���A�\Bp�H@fff@��@���B�HBX��                                    Bxv�Y  b          @�@��@�=q?s33@�{BS33@��@�@e�A�RB:                                    Bxv�g�  �          @�\)@��@\?��@�BZ��@��@��@UA�ffBGff                                    Bxv�vd  
�          @�(�@5�@�ff?���A��B���@5�@�=q@��\B
B|33                                    Bxv�
  !          @��@J=q@�p�?�=qA#�B�B�@J=q@���@��B�Bl                                      Bxvᓰ  T          @��@S33@�\)?�{A)p�B|  @S33@��H@��B�
Bd
=                                    Bxv�V  
�          @���@Q�@��?n{@�Q�B}�R@Q�@��@y��A�z�Bj{                                    Bxv��  T          @�ff@Dz�@�
=?n{@��B��
@Dz�@�  @}p�B ��BsG�                                    Bxvῢ  �          @��H@S�
@�G�?fff@ٙ�B�
=@S�
@��\@~{A��Bm\)                                    Bxv��H  y          @�Q�@n{@���>��
@Q�Bu  @n{@��H@\��A�Bf�                                    Bxv���  a          @���@U@�
=?�@}p�B~33@U@�ff@fffA�=qBnz�                                    Bxv��  "          @�=q�\@�33@;�A��B�Ǯ�\@W�@��\BW��B䞸                                    Bxv��:  �          @��H�˅@�Q�@C33A��\B�G��˅@AG�@���B_G�B�                                    Bxv��  	�          @�Q��G�@�{@A�A�
=B�  ��G�@>{@�\)B`��B���                                    Bxv��  
          @�����{@�ff@;�A��B�B���{@AG�@�z�BX(�B�Q�                                    Bxv�&,  a          @�z�8Q�@�ff@#�
A��Býq�8Q�@Z�H@�BU  B˽q                                    Bxv�4�  �          @��H�5@��H@.{A���B��H�5@P  @���B\=qB̞�                                    Bxv�Cx  T          @�\)�Q�@��@*=qA�BƳ3�Q�@L(�@�p�B[  B��)                                    Bxv�R  
�          @��
��\)@��@>�RA��HB�8R��\)@QG�@�=qB^�B�                                      Bxv�`�  "          @ʏ\�#�
@�33@5A�=qB�\�#�
@i��@��BX�B��f                                    Bxv�oj  
�          @�=q?aG�@�\)?޸RAw�B��?aG�@�\)@�p�B*z�B��f                                    Bxv�~  T          @�Q�@!�@�p�>\)?���B�ff@!�@��@>{A�\)B��)                                    Bxv⌶  �          @�Q�@fff@�Q�@  ��Bp\)@fff@���@{A�(�Bjp�                                    Bxv�\  "          @��@o\)@����
=�G�Bb
=@o\)@�=q?ǮAQG�B`G�                                    Bxv�  	�          @߮@U@��R��ff�,  Br@U@�z�?�=qAR�RBq��                                    Bxv⸨  �          @��H@E�@˅�\�C33B�W
@E�@�=q@%�A�\)Bx�\                                    Bxv��N  
�          @��@!�@��>��@r�\B���@!�@��
@Z�HA�Q�B��                                     Bxv���  T          @�{@ ��@߮�W
=���B�� @ ��@��@@��A�33B���                                    Bxv��  T          @�@   @�Q�?�@z�HB�#�@   @�
=@q�A�Q�B�G�                                    Bxv��@  T          @��?\@�{?�z�A4(�B�G�?\@���@�BB�=q                                    Bxv��  
(          @�@�@ۅ?E�@�G�B��@�@�  @tz�A�\)B�(�                                    Bxv��  �          @��
@P��@��H����   B~�R@P��@�\)@1G�A�G�Bu��                                    Bxv�2  
�          @���@HQ�@��>�ff@^�RB���@HQ�@�\)@[�A��
Bu\)                                    Bxv�-�  
�          @��H@E�@���  ��\)B�\@E�@���@;�A�B�                                    Bxv�<~  �          @�(�@#33@ᙚ?���A&�\B�8R@#33@���@�B{B�p�                                    Bxv�K$  
�          @�  @G�@��
?�Q�Ap��B�� @G�@��@�33B!\)B�=q                                    Bxv�Y�  �          @�G�?��H@�{@7
=A��\B�ff?��H@�p�@��\B?33B��                                    Bxv�hp  "          @�=q?�@أ�@333A�B�aH?�@���@��B={B�33                                    Bxv�w  
(          @�{?xQ�@�p�@eA�p�B��q?xQ�@��@ÅB]ffB�G�                                    Bxvㅼ  �          @�
=>�@�ff@�G�BQ�B��q>�@_\)@�G�Bs�B�G�                                    Bxv�b  "          @޸R>u@�(�@O\)A���B��{>u@�Q�@��BZz�B�=q                                    Bxv�  
�          @�p��(�@��R@hQ�B�
B��Ϳ(�@O\)@�ffBo�
B�ff                                    Bxv㱮  T          @��ÿ��H@�z�@S33A��BΊ=���H@G
=@�Q�Bfz�Bޣ�                                    Bxv��T  
�          @�{�Ǯ@��R@7�A���B�=q�Ǯ@e@���BX�HB��                                    Bxv���  "          @�ff?��\@�=q?��A>=qB�L�?��\@��H@x��B�B��                                     Bxv�ݠ  
�          @��H@Y��@�(��(�����Ba�@Y��@��?�G�A��B\�                                    Bxv��F  
�          @�  @�
=@i���޸R����B#�@�
=@�G�=��
?@  B/33                                    Bxv���  T          @�p�@��@�33�����4��B-(�@��@��R?B�\@�ffB/��                                    Bxv�	�  �          @�Q�@��\@<�Ϳ�=q��G�BQ�@��\@Vff��Q�p��B�R                                    Bxv�8  
�          @��@�p�@R�\�����P��A�Q�@�p�@h��=u>��HA���                                    Bxv�&�  "          @�  @�p�@<�Ϳ����\��A�ff@�p�@]p���{�"�\Aԏ\                                    Bxv�5�  
Z          @�ff@�?У��*=q���HAc\)@�@(�ÿ�z��V�RA��R                                    Bxv�D*  
�          @���@Ǯ?@  ��R��Q�@ڏ\@Ǯ?�\������A}��                                    Bxv�R�  �          @��H@��þ�z�� ����
=C��@���?\(��Q�����A�                                    Bxv�av  
�          @�  @�\)�L�Ϳ����~�\C�q@�\)?#�
��  �p  @�(�                                    Bxv�p  T          @��
@3�
@�33@7
=A�Q�Bo��@3�
@E�@��RBA�B>��                                    Bxv�~�  �          @�ff?��R@�G�@Q�A��B��?��R@^{@��
BD��B��=                                    Bxv�h  �          @�@C�
@�=q@Q�A���Bfp�@C�
@Q�@�Q�B.33B;�                                    Bxv�  �          @�{@   @��H@,��A���Bz��@   @J=q@��BBQ�BN��                                    Bxv䪴  "          @�
=@���?�ff@\)A�ffAJff@���=�\)@333A�33?+�                                    Bxv�Z  T          @�  @����#�
@��A�{C��R@����xQ�?�(�A�Q�C�=q                                    Bxv��   �          @�Q�@�p��B�\?\(�A	C�
=@�p����?0��@�z�C�=q                                    Bxv�֦  T          @�ff@��\�@  ?#�
@�\)C�(�@��\�u>u@�RC�"�                                    Bxv��L  �          @��H@�>��?�33A"{@g�@��\)?��HA*�HC�b�                                    Bxv���  "          @߮@�\)?u?�\)AV=qA@�\)>��?�\)Ax��?��                                    Bxv��  T          @���?5@�  ?ǮAg�B�z�?5@�
=@���B ��B��)                                    Bxv�>  �          @�z�?�@�\)?��
A��B��H?�@�(�@�33B(ffB�
=                                    Bxv��  �          @�p�?�@���?�ffA�(�B���?�@}p�@z�HB&��B�B�                                    Bxv�.�  T          @��H?�ff@�p�?�
=A���B��{?�ff@tz�@\)B-��B�k�                                    Bxv�=0  "          @�ff@-p�@��?��AQ��Bwff@-p�@~�R@Y��B
=B_                                      Bxv�K�  �          @�=q@�H@���?c�
A�B��f@�H@�z�@G
=A�Q�Btff                                    Bxv�Z|  T          @�  ?   @��
?�A�
=B�33?   @��@�ffB+B�33                                    Bxv�i"  
�          @��
�k�@ƸR@{A��B�Q�k�@�(�@�{B5G�B�Ǯ                                    Bxv�w�  "          @�Q�<#�
@��H?�\)A (�B�<#�
@�\)@p  Bz�B��3                                    Bxv�n  T          @�G��{@���=�G�?�G�B��H�{@�{@&ffA�=qB�W
                                    Bxv�  �          @���@  @���?��A��B�uÿ@  @�p�@��HB/�B�ff                                    Bxv壺  "          @��ÿB�\@��?�33Ab�RB�Ǯ�B�\@�33@g�Bz�B��
                                    Bxv�`  "          @�ff��Q�@�p�?�Q�A5G�B�녾�Q�@��\@g�B��B�u�                                    Bxv��  	�          @�=q����@��?�z�A��\B�.����@��@x��B'��B���                                    Bxv�Ϭ  
(          @�=q?��R@�(�?p��A�RB�Q�?��R@�@P��BQ�B���                                    Bxv��R  "          @�
=@/\)@�{>�@�=qB}{@/\)@�
=@0  Aՙ�Bop�                                    Bxv���  
�          @ȣ�@fff@�=q?(�@��BZ��@fff@��\@.{A�{BI
=                                    Bxv���  T          @Ǯ@z=q@��\>�z�@*�HBK�@z=q@��@z�A�p�B==q                                    Bxv�
D  �          @У�@��@�
=>���@_\)BFG�@��@��\@�RA�(�B6z�                                    Bxv��  
�          @�ff@�Q�@�z�=���?aG�BI�@�Q�@��@�A��B={                                    Bxv�'�  "          @ҏ\@\)@���k���(�BR�R@\)@��H@G�A��HBI�
                                    Bxv�66  �          @׮@�  @�(��\)���BU�\@�  @��?��
Au��BO��                                    Bxv�D�  
�          @��@H��@�p��5��Q�Bt
=@H��@�{?�G�Az�\Bo�H                                    Bxv�S�  T          @�{@�z�@���^�R��
=BS�H@�z�@��H?��
AL��BP�H                                    Bxv�b(  
�          @�
=@��\@Z�H���"=qA�  @��\@e>Ǯ@Tz�B(�                                    Bxv�p�  T          @�{@�Q�?�녿��H��33A@Q�@�Q�@���Q��=qA�                                      Bxv�t  "          @��
@Å@����(����A���@Å@@�׿E���
=Aљ�                                    Bxv�  T          @�=q@�p�@7����
�c�A���@�p�@W
=����Mp�A��H                                    Bxv��  
(          @���@���@l(����H���A���@���@vff>�@eB G�                                    Bxv�f  "          @ָR@3�
@�
=?Q�@�\B��\@3�
@�=q@P  A�{Bt
=                                    Bxv�  T          @�\)@,��@���?^�RA�B�
@,��@���@FffA�(�Boz�                                    Bxv�Ȳ  
�          @�(�?�
=@���@(�A�
=B��H?�
=@��@���B/�B�                                      Bxv��X  "          @���?Ǯ@��@�HA���B�u�?Ǯ@k�@��B>  B�#�                                    Bxv���  �          @��?��@�  @C�
A�G�B��?��@��@���B;��B��)                                    Bxv���  �          A ��?(��@�ff@_\)A�  B��?(��@�33@�p�BH�B�8R                                    Bxv�J  �          A   >�Q�@���@r�\A�z�B�.>�Q�@�=q@�(�BSffB��                                     Bxv��  "          A (�>��@أ�@�  B p�B��>��@�z�@�Bb�B�W
                                    Bxv� �  "          @�\)=�@�p�@��\B  B��f=�@���@ָRBf  B��R                                    Bxv�/<  �          @��׾��@�p�@�BQ�B����@k�@��Bu�RB£�                                    Bxv�=�  �          @���h��@���@�Q�B"  B�녿h��@Mp�@޸RB���B��
                                    Bxv�L�  �          @�33�{@�
=@:=qA�(�B����{@�\)@��
B8��B��                                    Bxv�[.  �          @��H� ��@���@W�A�z�B�#�� ��@�33@��BD�B���                                    Bxv�i�  �          @�\�@�\)@��B�B�.�@P  @��
Bc33B�k�                                    Bxv�xz  T          @�33�,��@��
@��RBffB�\)�,��@K�@ӅBf�
CO\                                    Bxv�   
�          @���3�
@�z�@��
B*
=B�L��3�
@�R@�Q�BwffC
��                                    Bxv��  T          @��R��@��@��BH33B��f��?�G�@��B�aHC.                                    Bxv�l  T          @�ff�{@�p�@�\)BI��B����{?��H@�{B��C�{                                    Bxv�  T          @�z��0��@u�@��BF��B��3�0��?�(�@�z�B�
=C+�                                    Bxv���  
�          @��E@���@�
=B+��B��H�E?�G�@��RBlC^�                                    Bxv��^  "          @�\)�-p�@�Q�@G�A�B��f�-p�@=p�@s�
B-�Cff                                    Bxv��  "          @�\)�P  @�(�@=p�A�B�#��P  @333@�\)B9
=CB�                                    Bxv���  �          @�����\@��H?5@�  C�����\@\��@33A��RC�                                     Bxv��P  
�          @�Q�����@��?�G�A��C�3����@E@S33B �C\                                    Bxv�
�  T          @�{���@���?��HA\��C	�����@E�@>{A���C�                                    Bxv��  T          @�����R@�(�?
=@��
C=q���R@�  @=qA�Q�Cp�                                    Bxv�(B  �          @�
=����@�G���Q��_\)C� ����@���>��R@@��C)                                    Bxv�6�  �          @��H�:=q@8Q������2��CW
�:=q@����#33���
B���                                    Bxv�E�  "          @�=q��ff@P�׿��
�B�RC���ff@`  >\)?��C�=                                    Bxv�T4  
�          @����@X�ýL�Ϳ�\C� ��@HQ�?��AJ�RC�H                                    Bxv�b�  
�          @�Q���  @�  ��\��33C  ��  @�녾����c33Cp�                                    Bxv�q�  �          @У���p�@QG��Y����C����p�@U?�@���Cc�                                    Bxv�&  "          @�����
@n{>�=q@�CG����
@S�
?�p�Au��Cc�                                    Bxv��  
�          @�p���G�@��׾L�Ϳ��HC� ��G�@p��?�AEC�=                                    Bxv�r  
�          @�����G�@p��?.{@��C�)��G�@J�H@�A�
=C0�                                    Bxv�  �          @�\)��Q�@]p�?���A?
=C�H��Q�@(Q�@(��A���C��                                    Bxv躾  
�          @ָR����@dz�?��A8��C)����@/\)@)��A�(�C�{                                    Bxv��d  
�          @��H��33@Fff?�z�AF�HC���33@�\@!�A�33C�=                                    Bxv��
  �          @���ff@>�R?��HAm��Cn��ff@�
@/\)A��HC &f                                    Bxv��  T          @�z����H@  @33A�{C
=���H?�G�@1G�A�z�C'G�                                    Bxv��V  
Z          @Ӆ��G�@$z�@{A�z�C(���G�?���@R�\A�G�C&�                                    Bxv��  �          @�\)����@)��?�=qAf�HCh�����?�@   A��C!ٚ                                    Bxv��  T          @Å����@j=q�.{��C�3����@hQ�?O\)@�{C�f                                    Bxv�!H  	.          @�33���@{��\�W�
C�=���@��R>\)?�p�C�
                                    Bxv�/�  T          @�����ff@p  �+���ffC5���ff@mp�?Y��@��
Cz�                                    Bxv�>�  "          @�33��@�녿}p����C33��@���?���A�CL�                                    Bxv�M:  	�          @����  @��\?��\@�C����  @�Q�@A�A�Q�C��                                    Bxv�[�  
�          @�\)��Q�@�{?^�R@�{CY���Q�@�{@4z�A�z�C��                                    Bxv�j�  �          @������\@��H?�(�A4z�C�����\@�33@[�AمCs3                                    Bxv�y,  "          @�\���H@�G�?���A.=qCn���H@��H@U�A���C
)                                    Bxv��  �          @׮�~{@��\�(���\)B�k��~{@���?�G�AP��B�W
                                    Bxv�x  
�          @�\�u�@���=�?�G�B�aH�u�@���@  A�z�B�L�                                    Bxv�  �          @�G�����@���?�p�A:�\C�)����@fff@G
=A�p�C{                                    Bxv��  �          @�33����@�\)?�33A1p�C#�����@e@AG�A�=qC#�                                    Bxv��j  �          @ָR��G�@�(���  �q��Cp���G�@�{>.{?��HC�=                                    Bxv��  "          @�\)��G�@��H�'
=��Q�C���G�@�Q�=p��ҏ\C ޸                                    Bxv�߶  T          @������H@�
=�
=q���C
����H@�
=�\�Mp�C��                                    Bxv��\  "          @��
���\@�p��C33�̸RC�����\@�Q쿏\)��C\                                    Bxv��  	�          @������@�33���p�C�����@��;����U�CQ�                                    Bxv��  
(          @�p����@�
=�����Q�Cٚ���@��þ�
=�U�C�H                                    Bxv�N  
(          A���\@0���G���Q�C���\@p  ��=q�Mp�C!H                                    Bxv�(�  �          A ����{@ff�<����C#�q��{@C�
��
=�]G�C�                                    Bxv�7�  
�          A���p�@   �7���p�C Ǯ��p�@Y���ٙ��A��C��                                    Bxv�F@  T          @��H�ָR@33�8������C#
=�ָR@?\)����g\)C�q                                    Bxv�T�  �          @�{��Q�@
=q�����C"�f��Q�@(������\C �\                                    Bxv�c�  T          @�����@3�
����"ffC�
���@G
=�.{��ffC                                    Bxv�r2  �          @���׮@:=q�У��H��C�)�׮@Tz����H��C��                                    Bxv��  �          @�\)�ٙ�@�Ϳ�p��W33C (��ٙ�@;��.{��
=C��                                    Bxv�~  
�          @��
��33@!��������C�q��33@I�����
��G�CQ�                                    Bxv�$  
�          @�R�ڏ\?��R�(����RC#�ڏ\@*�H���
�ffC��                                    Bxv��  �          @�z�����?��
��(��G�
C(�����?޸R�c�
��\)C%)                                    Bxv�p  "          @޸R�˅@?�  Ak
=C!�{�˅?��
@=qA���C(��                                    Bxv��  
�          @�\)��p�@-p�?�p�Af{CL���p�?�\)@'�A��RC#+�                                    Bxv�ؼ  �          @��
��  @1G�@=qA��CB���  ?�@QG�A��C#Ǯ                                    Bxv��b  
�          @���=q?��@��A���C#�q��=q?s33@,(�A�(�C+�q                                    Bxv��  �          @�=q��  ?У�@�RA�G�C%���  ?.{@+�A�(�C.                                      Bxv��  
Z          @������?�p�@�HA�=qC'B����>�@2�\A�=qC/�\                                    Bxv�T  �          @ٙ���z�?u?�ffAffC+Ǯ��z�?   ?�=qA4��C/�                                    Bxv�!�  "          @�33��p�?��
�L���߮C*�f��p�?�  ��Q��J=qC(��                                    Bxv�0�  
�          @Ϯ�ȣ�?�\)���&�RC)���ȣ�?�(��.{���C&�\                                    Bxv�?F  
�          @��H���?��
�����G�C*ٚ���?��Ϳ(����\C({                                    Bxv�M�  �          @����?5� ����  C-L����?�33��{�h  C&�R                                    Bxv�\�  
�          @�  �Å?��׿������C)���Å?޸R��=q�=�C${                                    Bxv�k8  T          @�(��˅?��׿����J=qC)��˅?�=q�n{���C&                                    Bxv�y�  
�          @���  ?��
�J=q�أ�C(�H��  ?�p�����G�C'33                                    Bxv누  T          @�
=��z�?n{�����XQ�C,  ��z�?�G����uC+T{                                    Bxv�*  
Z          @��
��G�?aG���G��r�\C,L���G�?z�H���
�:�HC+s3                                    Bxv��  �          @�����{?�=q<��
>B�\C*����{?}p�>��@g�C+8R                                    Bxv�v  T          @�G���{?�����ͿaG�C*aH��{?�ff>���@)��C*��                                    Bxv��  �          @У���?����Q�G�C*ٚ��?�  >�z�@$z�C+.                                    Bxv���  
�          @ָR��  ?��R��R��(�C)0���  ?��׾����C'�q                                    Bxv��h  
�          @�=q�ə�?p���=q����C+���ə�?�  ��z���  C$�                                    Bxv��  "          @��H�Å?z�H�����ffC*�)�Å?�p���  �v�RC$+�                                    Bxv���  �          @�Q����R?&ff�!G�����C-�����R?�G�����Q�C%Ǯ                                    Bxv�Z  
�          @�����>Ǯ�!�����C0\)����?��
�{��{C(G�                                    Bxv�   
�          @�G����\?#�
�4z���p�C-����\?�{�����p�C$��                                    Bxv�)�  T          @˅��G�>�(�� ����G�C/Ǯ��G�?�ff�(���Q�C'\)                                    Bxv�8L  �          @������\�0���n{��HC;Ǯ���\?.{�n�R���C,\)                                    Bxv�F�  
�          @�p���zᾔz��@�����C6�f��z�?O\)�:�H����C+�
                                    Bxv�U�  �          @�Q���\)�W
=�Dz��ᙚC6���\)?fff�<(���\)C+�                                    Bxv�d>  
�          @ȣ���녿+��P  ��=qC;:����?��Q����\C.W
                                    Bxv�r�  �          @�=q�����(��Y���  C:������?#�
�X���C-&f                                    Bxv쁊  �          @�G����
�0���L����z�C;O\���
>��H�N�R���C.�\                                    Bxv�0  T          @�z���33�}p�����'�HC@!H��33?�����+��C,�                                    Bxv��  "          @ڏ\�����
=��\)�1�CEG����>������;p�C/�                                    Bxv�|  �          @�z���=q���ff�$��CM�f��=q�
=q��\)�<p�C:Y�                                    Bxv�"  �          @�z����������p��ffCM�R���׿=p���  �233C<Y�                                    Bxv���  �          @�Q���p������  ��CJ����p��!G������$�RC:��                                    Bxv��n  �          @�=q��{�!G���p��\)CL�H��{�fff��G��$  C=h�                                    Bxv��  
�          @�����0  �z�H���HCM�����������(��(�C?ٚ                                    Bxv���  "          @��H����
=q�����Q�CI������R������
C:��                                    Bxv�`  
�          @���Q쿈����\)�G�C@���Q�>�����H�#(�C.�H                                    Bxv�  
�          @�Q����R��Q��c�
�G�CE�����R��{�{���HC7�                                    Bxv�"�  �          @�G����R�ff�hQ����CLE���R�p����
=�(�C>E                                    Bxv�1R  .          @����ff�C33�w���
CRaH��ff�\��{�#p�CDJ=                                    Bxv�?�  �          @�p������"�\�������CL.���Ϳ�ff�G���ffCCaH                                    Bxv�N�  �          @��������:=q�=q����CM�=���ÿ�33�QG����HCE�                                     Bxv�]D  "          @�Q����Z=q��H���
CQٚ���
=�\(����CI�R                                    Bxv�k�  
Z          @�  ��33�`���z���z�CT
=��33�   �XQ����HCL                                    Bxv�z�  T          @�\)��G��,���!���ffCN�q��G���z��S33���CExR                                    Bxv�6  T          @�����Q��
�H�0  ��p�CI���Q쿌���U���G�C?E                                    Bxv��  T          @��
�����+��#�
����CL�����׿���Tz���  CC��                                    Bxv���  �          @Ӆ���R�0���>{�ծCO�H���R��=q�n�R�	��CD�)                                    Bxv��(  "          @�ff���\�J�H������CR�����\�	���W
=��z�CI�q                                    Bxv���  	�          @������(���\�~{CK������Q��   ��33CDff                                    Bxv��t  	�          @У���Q���H����G�CI����Q����{�@��CG#�                                    Bxv��  "          @�G��o\)�#33�\��z�CVO\�o\)�����\��CN�                                    Bxv���  "          @�z��c�
�"�\�Ǯ����CWxR�c�
��{�z���Q�CO�)                                    Bxv��f  
�          @�z���33�E��=p����RCM�)��33�)����p��a�CJ�f                                    Bxv�  
Z          @����ƸR�R�\�aG���ffCO���ƸR�AG������,��CM�3                                    Bxv��  "          @�\)���
�l(���ff�/33CUG����
�B�\�p�����CPc�                                    Bxv�*X  
�          @�
=��
=��
=����G�CZ���
=�L(��g
=��Q�CSn                                    Bxv�8�  �          @�\)�����w
=���R�M�CX&f�����H���,(���p�CR��                                    Bxv�G�  �          @�\)���
�l(���Q��h��CV�����
�:�H�5��ƸRCP��                                    Bxv�VJ  �          @������=q��p��)��CZE����[��!G�����CU��                                    Bxv�d�  T          @������fff�����$  CY�����B�\������RCU�                                    Bxv�s�  �          @�p����|�Ϳ�  �Q�CZ� ���Y��������CV�{                                    Bxv�<  "          @����=q�L(����G�CR����=q��R�P����  CJ�=                                    Bxv��  �          @ҏ\����]p��-p���p�CWz�������l���	�CN�                                    Bxv  	�          @�  ������  �����G33C[�������Tz��(Q����CV��                                    Bxv�.  �          @�Q����R���ÿ�{�@��Cap����R�u��1G���{C\��                                    Bxv��  T          @�p��w����H���|��ChǮ�w����R�XQ�����Ccs3                                    Bxv��z  "          @���w��p  ������
C`��w��4z��Tz��
�CX
                                    Bxv��   �          @�{�e��\���\����CT�=�e��  ������ffCM��                                    Bxv���  �          @���~�R�zῑ��dQ�COs3�~�R��=q��  ��(�CI��                                    Bxv��l  T          @��������(��
=q���HCK������
=��z��Y�CH��                                    Bxv�  
�          @�Q���ff��H>��@�{CL�3��ff�(���{�Z�HCM#�                                    Bxv��  "          @������6ff���H��\)CMs3����"�\����A�CJ�3                                    Bxv�#^  T          @�
=�����<(���  �
�HCM�q�����,�Ϳ����$  CL�                                    Bxv�2  �          @��
��
=� ��=�\)?5CL���
=����=p���33CK��                                    Bxv�@�  T          @�����E?��@���CO�����G
=��(��o\)CO��                                    Bxv�OP  
�          @��H��33�333?�\)A��CM����33�@��>��?���CO@                                     Bxv�]�  "          @�z���{�*�H?��
A1��CL8R��{�<(�>�33@@��CNW
                                    Bxv�l�  
�          @�33���
��\?.{@���CJ33���
��ý�Q�^�RCK�                                    Bxv�{B  
�          @�z��ƸR���H?��A��CExR�ƸR�(�>�Q�@FffCGaH                                    Bxv��  
�          @ҏ\��33���?   @�(�CIp���33����z��#33CI�R                                    Bxv  T          @�=q��(��z�>B�\?�\)CH����(���׿\)��{CH0�                                    Bxv�4  T          @��H��\)�K�?\)@��HCQ���\)�L(����H��  CQ�                                    Bxv��  �          @��H�����C�
?(�@��\CO�������E�Ǯ�Z�HCP{                                    Bxv�Ā  
�          @�33��p��N{?�  A\)CQ����p��W���\)�#�
CR��                                    Bxv��&  �          @�33���P  ?c�
@�G�CQ�����W��8Q�˅CR��                                    Bxv���  T          @�33��ff�L(�?p��A�\CQ=q��ff�Tz����ffCR=q                                    Bxv��r  "          @�(���\)�N{?aG�@���CQT{��\)�U��8Q��ffCR.                                    Bxv��  "          @У����\�R�\?B�\@�\)CR�\���\�W
=���R�0  CS)                                    Bxv��  
(          @�  ����P��?@  @��
CR\)����U����R�1�CR��                                    Bxv�d  T          @����  �L��?��@�z�CR@ ��  �N�R��G��z�HCRs3                                    Bxv�+
  �          @�(���p��3�
?=p�@�z�CN^���p��9���8Q�У�CO
                                    Bxv�9�  �          @�������?J=q@�\CJ�����!G�    <�CK
                                    Bxv�HV  �          @�G����R�&ff?8Q�@˅CK�{���R�,�ͽ���=qCL\)                                    Bxv�V�  �          @�G���
=�!G�?W
=@�(�CJ�H��
=�)��<#�
=�G�CK��                                    Bxv�e�  T          @�\)���
�'�?J=q@�Q�CL
=���
�/\)�u�\)CL��                                    Bxv�tH  T          @�Q���\)��?}p�AQ�CIL���\)�!�>aG�?�CJ�H                                    Bxv���  "          @���ff�p�?��\A�\CHY���ff��H>�z�@%�CJ
                                    Bxv�  �          @�p����G�?fffA ��CH����(�>��?�\)CJT{                                    Bxv�:  �          @���Å�Ǯ?}p�A=qCBY��Å���>�@�33CDW
                                    Bxv��  �          @љ���녿��?���A
=C@ff��녿�z�?�R@�ffCB��                                    Bxv�  T          @�  ����33?�=qA��CB�q����33?�@��CE!H                                    Bxv��,  �          @�\)�Ǯ���?�ffA��C@���Ǯ��33?z�@�33CB�                                    Bxv���  �          @θR�ƸR���?�z�A%��C?�f�ƸR����?333@�  CBu�                                    Bxv��x  
�          @�33��  ��?z�HA�RCDǮ��  �G�>\@^{CF��                                    Bxv��  
(          @ə���{��(�?�z�A*=qCD&f��{���R?z�@��CF��                                    Bxv��  
�          @ʏ\��{��ff?���A/
=CD�)��{��?
=@���CGL�                                    Bxv�j  
�          @��������?�z�A)G�CE\����?��@�Q�CGaH                                    Bxv�$  �          @ȣ���33��p�?��HA}p�CB5���33��
=?�
=A-�CFE                                    Bxv�2�  �          @�
=��ff��G�?�  A��\CE5���ff�p�?���A'�CI33                                    Bxv�A\  
�          @�p���=q��?�A��\CF�q��=q�Q�?�A/33CK(�                                    Bxv�P  �          @�p���G���{@   A���CF����G���?��AI�CK33                                    Bxv�^�  
�          @\��z���
?�A�CC.��z���H?��A,(�CG0�                                    Bxv�mN  "          @�33���Ϳ�z�?�A�ffCB���Ϳ��?��AE�CF�                                     Bxv�{�  
�          @���  ��@�
A���CG:���  ���?��AO�
CK�q                                    Bxv�  "          @Ǯ��33��
@��A���CKL���33�5?���AH��CO�                                    Bxv�@  T          @�Q���(��Dz�@(�A��CT+���(��i��?�z�AP��CXٚ                                    Bxv��  
�          @�p���\)�1�@{A�\)CO�q��\)�X��?��
A]G�CT��                                    Bxv�  
�          @Ǯ�����Q�@&ffA�=qCCaH����	��@33A�33CJJ=                                    Bxv��2  	�          @�  ���R����@=qA�=qC70����R��G�@p�A��C>p�                                    Bxv���  "          @����G���@!�A�ffC9ff��G�����@G�A���CAn                                    Bxv��~  �          @���K�@N�R@
=A�(�C���K�@p�@>{B��CQ�                                    Bxv��$  �          @��R?�(�@�Q�B�\���
B���?�(�@�\)?xQ�A�RB��)                                    Bxv���  "          @�?�p�@�  �W
=��
=B�Ǯ?�p�@�
=?��AG�B���                                    Bxv�p  �          @��
?���@�p��@  ����B��H?���@��
?��AffB��                                    Bxv�  
�          @ə�?n{@�{�(�����B��H?n{@�33?��RA5G�B���                                    Bxv�+�  T          @�
=?u@�(���ff���B�(�?u@��?�\)AK
=B�                                    Bxv�:b  T          @Ǯ?@  @�p��#�
��
=B�{?@  @�ff?�33Av�RB��\                                    Bxv�I  
�          @Ǯ?�=q@\>8Q�?�B�ff?�=q@���?���A�B��                                    Bxv�W�  �          @�  ?�33@�33?!G�@�=qB���?�33@���@Q�A�\)B���                                    Bxv�fT  �          @�Q�?aG�@��H?��A (�B��{?aG�@�Q�@3�
AծB��H                                    Bxv�t�  
�          @\>�@���?�{AO\)B�z�>�@�  @@��A�ffB�(�                                    Bxv�  "          @\=�@�(�?�
=A[�B��{=�@��R@Dz�A�(�B�G�                                    Bxv�F  �          @Å?.{@�Q�?�{A�=qB�z�?.{@�\)@\��B	�RB�aH                                    Bxv��  T          @�(�=�G�@���?��HA�z�B��q=�G�@�ff@b�\B33B�\)                                    Bxv�  
�          @�33��=q@�=q?�\)Ax(�B�Q쾊=q@��@N{B ��B�\                                    Bxv�8  T          @�G�>�p�@��@ffA��HB�\)>�p�@��@eB��B�                                      Bxv���  
�          @�����@�{@ ��A�\)B�
=���@���@UB(�B�.                                    Bxv�ۄ  
          @���?�(�@[�@l(�B5ffB�
=?�(�@\)@���Br\)Bu��                                    Bxv��*  `          @���@��@5�@�=qBTffBM=q@��?�=q@�B�\)A��
                                    Bxv���  T          @�z�@Q�@1G�@�G�B\�BQ��@Q�?�(�@��
B�A�R                                    Bxv�v  
(          @��@
=@X��@�{BH��B\��@
=?�@�ffByG�B                                      Bxv�  "          @�
=@N�R@5@��B>��B%33@N�R?���@�33Bb�RA��H                                    Bxv�$�  �          @˅?��@�=q@��\B-�
B�W
?��@0  @�=qBj�RB|�
                                    Bxv�3h  �          @����p�@�  @\)A��B�녿�p�@��@�Q�B�B��                                    Bxv�B  
�          @�  �r�\@�z�>�{@J=qB���r�\@�33?��HA�
C}q                                    Bxv�P�  
�          @љ����@���@<��A�z�B��)���@��R@�(�B(p�B���                                    Bxv�_Z  �          @�
=��
=@�Q�@HQ�A�=qB�녿�
=@���@��
B.{B�8R                                    Bxv�n   
�          @�  ����@�G�@E�A���B�\)����@��R@�Q�B0ffBճ3                                    Bxv�|�  �          @Ӆ���@��@L(�A�B��)���@��@�(�B2�RB�33                                    Bxv�L  T          @Ӆ���@�(�@��B��B�Q���@s33@��\BX��B��                                    Bxv��  
�          @�=q��G�@��
@s33B�
B�녿�G�@w
=@��\BK33B��
                                    Bxv�  �          @׮����@��\@|(�B��B�녿���@q�@�ffBJ��B�u�                                    Bxv�>  H          @�\)��G�@�G�@{�BG�B��H��G�@\)@��BM�B�#�                                    Bxv���  
f          @�ff��p�@���@S33A��B��
��p�@�z�@��RB3��B�{                                    Bxv�Ԋ  �          @�
=�8Q�@��R@r�\B
�B�\�8Q�@�ff@���BI�Bǀ                                     Bxv��0  
�          @�z���R@�
=@1�Aƣ�B�����R@��@�BB��H                                    Bxv���  
�          @�p����H@�z�@:=qA�(�B�G����H@��
@��B#�B���                                    Bxv� |  T          @�
=���@Ǯ@
=A�\)B��)���@�p�@p  BG�B�W
                                    Bxv�"  H          @�(���\)@���@s�
BG�B��Ϳ�\)@��@��
BJQ�B��
                                    Bxv��  
4          @�Q�>k�@n{@�
=Blp�B��
>k�?�\)@�  B��B���                                    Bxv�,n            @�=#�
@k�@�  Br
=B��=#�
?�G�@�Q�B�ǮB�=q                                    Bxv�;  `          @�\)�\@�ff@|(�A���B���\@�(�@���B<��B��\                                    Bxv�I�  	�          @����(Q�@���?&ff@���B۸R�(Q�@ڏ\@'
=A��B�#�                                    Bxv�X`  
�          @�(���Q�@�G�?k�@�z�B��)��Q�@���@333A���B�(�                                    Bxv�g  
�          @陚���@�G�?��A	��B�����@Ǯ@8��A�33B�\                                    Bxv�u�  
�          @�\�(Q�@�G�?B�\@�p�B�W
�(Q�@ʏ\@$z�A�=qB��                                    Bxv�R  �          @�33�H��@��
>���@#�
B����H��@�G�@
=A��B�#�                                    Bxv���  �          @��2�\@�\)?:�H@�ffB�
=�2�\@���@!G�A���B��H                                    Bxv���  "          @�ff����@�
=?�Ae��B�G�����@ƸR@i��A�p�BԨ�                                    Bxv��D  H          @�  �8Q�@ҏ\?�(�AffB�Q��8Q�@�Q�@<��A��HB�G�                                    Bxv���  
�          @�p��J=q@�z�?�(�AP��B�(��J=q@��@h��A�(�B��                                     Bxv�͐  
�          @�(���@��H?�\)Ac�
B��=��@�=q@qG�A�{B�k�                                    Bxv��6  
�          @�\)�c�
@��
?�z�An�\B�LͿc�
@�33@p  A�ffB�                                    Bxv���  �          @�Q��<(�@У�@7�A��RB�u��<(�@���@�  B�B�                                    Bxv���  H          @�
=�Fff@�{@(�A���B����Fff@��@y��A�B��
                                    Bxv�(  
e          @�  �33@���?(�@��\B���33@�
=@#�
A�p�BԽq                                    Bxv��  
�          @�Q쿇�@�z�(���{B�Q쿇�@���?�\)A(��BĊ=                                    Bxv�%t  �          @�(���33@��
>�G�@U�B���33@�\)@Q�A���B�u�                                    Bxv�4  
�          @��H�	��@�Q�k���(�B�  �	��@��?ٙ�AO�
B��)                                    Bxv�B�  "          @��H�33@ᙚ��(��3\)B���33@�{>�ff@X��B�u�                                    Bxv�Qf  
(          @陚�33@�(���33�r=qB�L��33@�zὣ�
�&ffB��                                    Bxv�`  
�          @����@�G��  ��=qB�LͿ��@��;�ff�e�Bҳ3                                    Bxv�n�  
�          @�\)�G�@�Q��*=q���
B�p��G�@Ϯ�}p��ffBڔ{                                    Bxv�}X  "          @�ff��ff@����p����
B��þ�ff@�Q��ff�s33B��\                                    Bxv���  
�          @޸R?
=@�
==�?}p�B��?
=@θR?�\)A��HB��                                     Bxv���  T          @�ff��ff@����ff�fffB�.��ff@���?�Q�A8Q�B�Q�                                    Bxv��J  T          @��;�p�@�\�z���z�B�𤾽p�@߮?��A'�B�                                    Bxv���  �          @�33�aG�@�׾�p��?\)B�W
�aG�@��
?�(�A?�
B£�                                    Bxv�Ɩ  T          @�=q���
@޸R��Q��<��B��쿃�
@�=q?��HA?\)B�.                                    Bxv��<  �          @�ff��=q@�33�}p��
=B�p���=q@��
?Tz�@�33B�p�                                    Bxv���  T          @���>�=q@�
=��ff�.�RB�\)>�=q@�=q>��H@��HB�p�                                    Bxv��  �          @ۅ��@љ��޸R�l��B�B���@أ�<��
>\)B���                                    Bxv�.  "          @�녾�z�@�z������B�(���z�@�\)����aG�B��f                                    Bxv��  
�          @��H��{@θR�.{���HB����{@�G�?�  AN=qB���                                    Bxv�z  T          @߮����@Ӆ?���A�B�=q����@�33@333A���BոR                                    Bxv�-   
�          @�{���H@ָR�#�
��G�B�zῺ�H@���?�{A(�B̳3                                    Bxv�;�  
�          @�  ���@ƸR���H�,Q�B�.���@�=q>�
=@o\)B�Ǯ                                    Bxv�Jl  �          @��;�p�@˅>k�@�B�����p�@��H?���A��B��                                    Bxv�Y  "          @�Q쾞�R@�\)=L��>�(�B������R@�Q�?��As�B��)                                    Bxv�g�  	�          @Ǯ��@�ff>�z�@,(�B�Q��@�p�?�{A�B�p�                                    Bxv�v^  �          @�G��fff@Å?}p�A�BĽq�fff@��@ ��A�G�B�\                                    Bxv��  T          @ə��n{@�\)?У�Aq�BŽq�n{@��
@FffA�G�BǸR                                    Bxv���  �          @ə����@���?���Amp�B��
���@�p�@Dz�A�{B��R                                    Bxv��P  T          @�p�����@�\)@�A��B�aH����@�  @c�
Bp�B�B�                                    Bxv���  "          @�33�u@ȣ�>�?���B���u@���?��HA}B�#�                                    Bxv���  
�          @ʏ\?�\)@���  ��z�B���?�\)@�{����(�B��{                                    Bxv��B  -          @�33?\@���>{��  B�{?\@�z���
�bffB�{                                    Bxv���  �          @У�>\)@�ff�����z�B�aH>\)@ə��z���\)B��                                    Bxv��  "          @˅?p��@��H��p���ffB��R?p��@\�W
=��B�p�                                    Bxv��4  
�          @�  ��\)@�=q@/\)A�\)Bӽq��\)@~�R@w
=B*33B�                                      Bxv��  �          @��H��  @��\?˅A}��B�\��  @�Q�@<(�A�RB���                                    Bxv��  �          @��R>�33@��H?xQ�A33B�#�>�33@��@��A�G�B���                                    Bxv�&&  T          @ƸR?�z�@��ͽ��Ϳn{B��?�z�@��?���AM�B�#�                                    Bxv�4�  �          @�p�<��
@��?k�A	B���<��
@�z�@��A�ffB��{                                    Bxv�Cr  �          @�녾�33@��?���A-�B���33@�{@$z�Aə�B�W
                                    Bxv�R  "          @��ÿ��@�\)?��@�(�BȞ����@��?���A�(�B�Ǯ                                    Bxv�`�  �          @ƸR���R@���?G�@�Q�B����R@��@(�A�ffB�p�                                    Bxv�od  �          @����@�z�@3�
A���B�Ǯ��@�G�@|(�B {B��
                                    Bxv�~
  "          @���@��@X��B�
B�\��@dz�@�p�B7G�B�33                                    Bxv���  �          @�
=�
=@~{@��B)��B�\�
=@5�@��RBZQ�B��                                     Bxv��V  �          @���  @��?ǮAk
=B�aH��  @���@=p�A��B�ff                                    Bxv���  �          @��ÿ�@��R?�A�  B�uÿ�@��H@J=qA�p�B��R                                    Bxv���  T          @�  �W
=@��\?W
=Az�B�ff�W
=@�ff@�A�  B��R                                    Bxv��H  �          @��R?+�@�=q?(�@�B�aH?+�@�  ?��HA�=qB��\                                    Bxv���  �          @���?�z�@��?��A��\B�  ?�z�@�  @E�B�B�p�                                    Bxv��  �          @��
@p�@��\>\@fffB�k�@p�@�=q?޸RA�(�B�W
                                    Bxv��:  �          @ȣ�@,(�@��?��AA�B�\@,(�@�Q�@'�AŮBv�                                    Bxv��  �          @�z�@!�@�  ?�ffA(�B�� @!�@�=q@�A��HB��                                    Bxv��  �          @ə�@n�R@�Q�?8Q�@�=qBU=q@n�R@�?�A�{BM                                    Bxv�,  �          @�Q�@>{@��
?�\)A%p�Bt
=@>{@�@��A�Q�Bk�                                    Bxv�-�  "          @ə�@=q@�
=?��@�  B�=q@=q@���?�(�A�33B��)                                    Bxv�<x  �          @Ǯ?���@�G���G���B��
?���@Å>��H@�  B�\                                    Bxv�K  T          @�33?�Q�@��\����L(�B�G�?�Q�@�  =u?\)B�8R                                    Bxv�Y�  
�          @���@s33@�  �7
=��33B:  @s33@�=q����G�BI33                                    Bxv�hj  �          @���@"�\@����R���
B|{@"�\@�(��u��HB�u�                                    Bxv�w  "          @�ff@HQ�@g��qG����BD�R@HQ�@��2�\��BZ�                                    Bxv���  "          @�ff@7
=@�\)�U��Q�B_�R@7
=@���p���(�Bo                                      Bxv��\  �          @���?��@�  �z�H�
=B��R?��@�=q>�(�@��HB�                                    Bxv��  �          @��
�0  @\)@j�HB��B�#��0  @?\)@���B@=qC�{                                    Bxv���  
�          @��?!G�@\?L��@�B�B�?!G�@�
=@�RA�B��                                    Bxv��N  �          @�{?.{@�
=?�
=A*�\B��?.{@�Q�@(Q�A�\)B��                                    Bxv���  T          @�z�>��R@��@:=qA�B��>��R@���@�B${B��\                                    Bxv�ݚ  T          @�(�>��@��R@hQ�B{B���>��@�{@���BB
=B��                                    Bxv��@  T          @�(��#�
@�(�@�33B,�RB���#�
@]p�@��\Bc�
B�L�                                    Bxv���  �          @�=q<#�
@�=q@mp�B��B��H<#�
@�G�@��BG�
B��
                                    Bxv�	�  �          @ə���  @��R@?\)A�G�B�
=��  @�33@�z�B)�
B��                                    Bxv�2  T          @˅��33@�p�@p��B(�BӨ���33@x��@�=qBE�B�z�                                    Bxv�&�  "          @ʏ\�p�@N�R@��
BL�HB����p�?�p�@��Bvz�C
#�                                    Bxv�5~  
�          @�����@k�@��B9��B�����@�R@�33Be=qC�3                                    Bxv�D$  {          @�p��5�@c�
@�{B1�C ���5�@=q@���BX�C�)                                    Bxv�R�  -          @�{��\@��H@�
=B%ffB�G���\@N{@�(�BUz�B���                                    Bxv�ap  "          @��  @���@l(�B{B��  @p��@��RB<Q�B�                                    Bxv�p  �          @�(���@�{@c�
Bz�B�W
��@}p�@�(�B:�B�=q                                    Bxv�~�  �          @ʏ\���H@�{@J�HA���B؊=���H@���@���B+��B�z�                                    Bxv��b  �          @�녿W
=@�ff?�
=A�33B�Q�W
=@��@6ffA�ffB�ff                                    Bxv��  I          @��H�G�@��@%A�{BĊ=�G�@��@qG�B��B�33                                    Bxv���  
�          @��H����@�G�>k�@��B�aH����@��?ٙ�Az�RB��{                                    Bxv��T  �          @ə�����@Ǯ?+�@��B������@��@
=A�\)B�u�                                    Bxv���  T          @˅>�
=@ȣ׾���p  B�k�>�
=@�{?��A�
B�Q�                                    Bxv�֠  
�          @�z�=���@�ff��33�L(�B�8R=���@˅=��
?:�HB�B�                                    Bxv��F  �          @�G�?���@�=q���,z�B���?���@�>�=q@   B�L�                                    Bxv���  "          @ʏ\���
@�  �z�����B�k����
@�
=?c�
A�\B�k�                                    Bxv��  
�          @ƸR=�\)@��׿��/�B���=�\)@�(�>�=q@�RB���                                    Bxv�8  
�          @�(��(��@�녿����p�B�p��(��@���?^�RA
=B��                                    Bxv��  "          @ə�>��@�Q�?��RA^=qB�B�>��@�  @6ffA�=qB��=                                    Bxv�.�  �          @�33?c�
@�@b�\B�HB�k�?c�
@�
=@���B<�B�{                                    Bxv�=*  
�          @˅>���@�(�@UA��B��\>���@��R@�  B4Q�B���                                    Bxv�K�  T          @��?�33@�Q�@L(�A�
=B�L�?�33@�(�@�=qB.�B��\                                    Bxv�Zv  �          @ƸR?��H@�z�@L��A�p�B��\?��H@���@���B0p�B�aH                                    Bxv�i  
�          @�p�?��R@��\@5�A��B��
?��R@���@\)B!G�B�p�                                    Bxv�w�  
�          @�p�>��
@�(�@6ffA�33B��>��
@�=q@�Q�B$B��                                    Bxv��h  "          @�z῱�@��@g
=B�BԽq���@q�@�33BC{B�k�                                    Bxv��  "          @ƸR�Y��@�33@9��A�G�B��Y��@�G�@���B%
=B�(�                                    Bxv���  T          @Ǯ�@  @�@�HA��B���@  @�
=@j=qB(�B��                                    Bxv��Z  T          @�ff���@��H?��HA�{B�#׾��@�\)@P  A�B�33                                    Bxv��   {          @�{�\)@��H?��RA�ffB���\)@�\)@Q�B (�B�#�                                    Bxv�Ϧ  
3          @�G�>�(�@�ff?��HA��
B���>�(�@��H@QG�A�\)B��R                                    Bxv��L  T          @�G�?h��@��@�A�
=B�B�?h��@��@S�
A�\)B�33                                    Bxv���  �          @��@
=@�z��U���RB��\@
=@����Q���ffB���                                    Bxv���  T          @�?�Q�@��\�Z=q� G�B�{?�Q�@�
=�
=q��G�B�u�                                    Bxv�
>  �          @�(�?�ff@�z��U����B�aH?�ff@�Q��z�����B�L�                                    Bxv��  �          @�z�@��@���<(���p�B��@��@��ÿ��r�\B�B�                                    Bxv�'�  "          @��
@��@�(���33����B�33@��@�p�����33B��                                    Bxv�60  T          @�G�?z�@�Q��*�H��=qB�
=?z�@�
=�����K\)B���                                    Bxv�D�  "          @ʏ\@Vff@�  ?!G�@���Be�H@Vff@�
=?�A�Q�B`�                                    Bxv�S|  
�          @�Q�@P  @�{?�p�A6=qBg��@P  @���@Q�A��B_
=                                    Bxv�b"  	`          @�ff@X��@�
=?�A��\BYz�@X��@�@7
=A�{BL{                                    Bxv�p�  
�          @���?��R@�33����z�B�(�?��R@�\)?��HA:=qB���                                    Bxv�n  "          @Å?�@�ff�����p��B���?�@�z�?uA�B��{                                    Bxv��  T          @�z�?��H@�z῝p��8z�B�Ǯ?��H@�Q�>\)?��\B�=q                                    Bxv���  "          @�p�?&ff@��
��Q��33B�L�?&ff@�33���
�>�RB�                                    Bxv��`  �          @���?�ff@�p���=q���B�{?�ff@�{�   ��
=B�33                                    Bxv��  T          @���>��
@�33��(���
=B�u�>��
@��Ϳ(�����
B���                                    Bxv�Ȭ  T          @���?}p�@�����R��33B���?}p�@�ff����QB�\)                                    Bxv��R  
�          @��?��@}p��tz��${B���?��@��R�5����HB�33                                    Bxv���  
�          @�z�>#�
@p��� �����HB��>#�
@����{�r{B��=                                    Bxv���  �          @�����p�?�p�@�33B�{C
=��p�>8Q�@�G�B�#�C-
=                                    Bxv�D  
�          @�p���  @_\)@�{BE�\B�{��  @=q@��Bs�B���                                    Bxv��  "          @���>���@�{@��A�\)B���>���@�Q�@g
=BQ�B�aH                                    Bxv� �  T          @Ǯ@
=@}p�@��B/\)Bw�R@
=@8��@��\B[ffBW�                                    Bxv�/6  �          @�  ?�@g�@��BE�HB}��?�@{@�BrG�BW
=                                    Bxv�=�  �          @��H@��@�\)@\)B�RBz
=@��@N�R@��HBL=qB^�                                    Bxv�L�  
�          @���@�
@�  @��A�G�B�u�@�
@�=q@dz�BffB|��                                    Bxv�[(  
�          @�G�@C�
@��\?��
A<��Bp@C�
@��@(�A�33Bhp�                                    Bxv�i�  "          @�{@z�@�  ?�@���B�\)@z�@�\)?�A�G�B��\                                    Bxv�xt  T          @�33@{@�논#�
���
B�#�@{@�p�?�p�A;
=B�\                                    Bxv��  
�          @��?�Q�@�����Q�Y��B�� ?�Q�@�?�33A4��B��3                                    Bxv���  
�          @�\)@2�\@�=q=#�
>��Bt�H@2�\@�{?�A:�\Br33                                    Bxv��f  
(          @��?���@�녽�G���=qB���?���@�{?���A8Q�B��                                     Bxv��  I          @�Q�?��@��>k�@B��?��@��
?��HAm��B�                                    Bxv���  
�          @�33?8Q�@�
=�8Q��\B���?8Q�@��
?���A/�B�W
                                    Bxv��X  �          @���?�(�@�ff�G��   B��=?�(�@��>Ǯ@�Q�B��)                                    Bxv���  "          @�G�?:�H@�33>u@�HB�33?:�H@��?�p�Ar=qB��                                    Bxv���  
�          @�
=@  @��R��������B�B�@  @�p�?E�@��B��                                    Bxv��J  T          @�G�?�{@�G��#�
�˅B��3?�{@�{?���A+\)B��                                    Bxv�
�  T          @��\?�
=@�G��8Q���B�=q?�
=@��?�@�z�B�aH                                    Bxv��  �          @���@p�@�p��(�����HB�@p�@�{?   @�
=B�.                                    Bxv�(<  T          @�@H��@��;���'�B_��@H��@��H?G�@�
=B^p�                                    Bxv�6�  T          @�
=@�G�@q녾�����B%�@�G�@n{?0��@�(�B#�R                                    Bxv�E�  
�          @�=q@�\)@S33�L�;�Bff@�\)@N�R?.{@��B                                      Bxv�T.  
�          @�ff@   @���?@  @�=qB~�
@   @�Q�?���A��By\)                                    Bxv�b�  �          @�ff@33@���?�@�  B��q@33@��?�
=A���B��H                                    Bxv�qz  
�          @��R?�\)@�p�?�\@�ffB���?�\)@�p�?ٙ�A��B�#�                                    Bxv��   �          @���?���@�z�>Ǯ@��B��)?���@�p�?���A���B��                                    Bxv���  �          @�=q?��@�ff>k�@��B�u�?��@�Q�?�
=An{B��                                    Bxv��l  "          @�  ?�  @��
�u�+�B���?�  @�  ?�\)A=p�B�\)                                    Bxv��  T          @���33@�(�?��HA|��B��
��33@�@#33A�  B��{                                    Bxv���            @���>�
=@��
?�\)A�B�33>�
=@��H@9��B�\B�                                      Bxv��^  
�          @��?#�
@��\?�(�A��RB�.?#�
@y��@��A�33B�\)                                    Bxv��  "          @��?n{@�p�@_\)B  B��)?n{@S33@�=qBNQ�B��                                    Bxv��  �          @��R?:�H@���@(�A��HB�k�?:�H@�z�@_\)B��B���                                    Bxv��P  	�          @�=q?&ff@��?�A���B��?&ff@�  @1�A�\)B��                                     Bxv��  "          @��H>u@�
=?�\@��B�(�>u@�
=?�  A��B��                                    Bxv��  �          @��>#�
@��\��33���B�aH>#�
@��
�:�H����B��{                                    Bxv�!B  
�          @�zῘQ�@�������B�aH��Q�@�
=?333@ۅB�p�                                    Bxv�/�  
�          @�(�>��@��?!G�@�  B���>��@��R?�A�G�B�ff                                    Bxv�>�  
�          @�녿B�\@��?c�
A�B�.�B�\@���@�\A��HB�8R                                    Bxv�M4  
(          @�ff����@�33?�G�A$��BԞ�����@�  @
=qA�p�B�Ǯ                                    Bxv�[�  
Z          @��׿��@���?W
=A
=BظR���@��\@ ��A�p�B��
                                    Bxv�j�  
�          @����@�  ?��@У�B�=q��@�  ?У�A��RB�3                                    Bxv�y&  "          @��H��p�?��R@$z�A�z�C#+���p�?c�
@5�A�=qC)                                    Bxv���  T          @\���?��H@z�A�\)C'�����?+�@!�Aď\C-                                      Bxv��r  �          @��
����?�\)?�Q�A�G�C"�����?�33@  A�
=C'��                                    Bxv��  "          @��\��?��?��\AI�C#:���?���?���A33C&\)                                    Bxv���  S          @��
���
?��H?�ffAv=qC"aH���
?�=q?��A���C&#�                                    Bxv��d  "          @�p����R?޸R?�{AU�C"Y����R?�33?ٙ�A��\C%��                                    Bxv��
  �          @�p�����@   >W
=@
=C:�����?�33?#�
@У�C !H                                    Bxv�߰  T          @�p���ff?�p�?�ffA$(�C ���ff?��H?��HAeG�C"�H                                    Bxv��V  �          @�����z�?��\?�{A�G�C&����z�?c�
?���A�33C*�H                                    Bxv���  T          @��H���R?(��?�ffA|  C-����R>��R?�z�A���C0��                                    Bxv��  
Z          @�������@��?�{Ay�C�\����?�p�@ffA�p�C^�                                    Bxv�H  �          @�����z�@�\?˅Av�RCB���z�?��@   A�(�C#
=                                    Bxv�(�  
�          @�=q���@�
?�33A/\)C�����?�G�?���Aqp�C"h�                                    Bxv�7�  
Z          @�33���?���?Tz�A�HC .���?�(�?�p�AD  C"Y�                                    Bxv�F:  "          @�=q�Q�@���?�@��HB�k��Q�@�p�?��AyG�B�(�                                    Bxv�T�  
�          @�\)�e�@�G�?z�HA�HC�
�e�@~�R?��A��C                                    Bxv�c�  �          @��R�\(�@�33?���A8Q�C W
�\(�@\)@33A�=qC��                                    Bxv�r,  
�          @�  �P��@��?(��@�p�B��=�P��@�33?�33A��B���                                    Bxv���  	�          @�\)�^�R@�G�>\)?���B��
�^�R@���?���A5�C J=                                    Bxv��x  
�          @�Q��J�H@�������%B���J�H@��?J=q@�B���                                    Bxv��  �          @�\)�<��@�33�.{�ڏ\B���<��@�(�>���@~�RB�Q�                                    Bxv���  �          @��\�!G�@���?��A=��B�k��!G�@�@
=qA��B�=q                                    Bxv��j  �          @�=q�=p�@�G�?���Af�\B�u��=p�@�33@p�A�{B�ff                                    Bxv��  T          @�(��C�
@��R?�R@��HB�p��C�
@�ff?�Q�A�{B�.                                    Bxv�ض  �          @�33�33@����u�B�W
�33@�(�>aG�@�B�                                    Bxv��\  T          @�{��@��
��33�<(�B�=q��@�      �#�
B�B�                                    Bxv��  T          @���>�R@���fff�(�B�8R�>�R@�  >\)?�B�G�                                    Bxw �  �          @����E�@�(�>��@�z�B�8R�E�@�{?�{Aj�\B��3                                    Bxw N  �          @��R�1�@��H?�@���B�ff�1�@��?ǮA�=qB��                                    Bxw !�  �          @����{@�Q�?��
A5�B�ff�{@�?��RA�z�B�.                                    Bxw 0�  �          @�(�� ��@��?aG�A{B��)� ��@��?�A�  B�k�                                    Bxw ?@  �          @�z��(��@��
>\)?�=qB�G��(��@�\)?��AEB��f                                    Bxw M�  �          @��\�P��@��
?333@�(�C ff�P��@w
=?˅A�p�C.                                    Bxw \�  �          @���G�@b�\?�=qA��HC
Ǯ��G�@G
=@�
A�G�Cn                                    Bxw k2  �          @�33���\@Mp�?�\)A�
=C�����\@1�@�A��\C�                                    Bxw y�  �          @�{��=q@I��?z�HA (�Cp���=q@7
=?�33A�C��                                    Bxw �~  �          @�ff���@=p��L�;�ffC�����@9��?��@��HC33                                    Bxw �$  T          @�\)��@�>�?���Cٚ��@?(��@��C��                                    Bxw ��  �          @�ff���@#�
>�
=@�
=C����@=q?xQ�AG�C��                                    Bxw �p  �          @�
=��33@#�
>�(�@���CY���33@��?z�HA�C                                    Bxw �  �          @�
=���@:�H>�=q@,(�CO\���@2�\?fffA��Cp�                                    Bxw Ѽ  T          @�{���@l(���33�c�
Cs3���@j�H>�@�C�=                                    Bxw �b  �          @�33��p�@mp���Q�fffC
\)��p�@h��?:�H@�C
��                                    Bxw �  �          @����Z=q@~�R��{�n�RC���Z=q@|��?
=q@�z�CǮ                                    Bxw ��  �          @����C33@��׾�33�tz�B�  �C33@��?��@�Q�B�\)                                    BxwT  �          @�{�z�@�=q�@  ��ffB�k��z�@��>���@_\)B���                                    Bxw�  �          @�\)�
=@��þ�
=����B�ff�
=@�  ?333@��B��                                    Bxw)�  �          @�=q�\)@�G���G���G�B��
�\)@���?.{@�=qB�{                                    Bxw8F  �          @���@��\?!G�@��
B�
=��@�=q?�p�A���B�ff                                    BxwF�  �          @����5�@���>�=q@(��B�q�5�@��H?�{AZ�RB��=                                    BxwU�  �          @�  �*�H@��R?���A5��B�=�*�H@��H@(�A���B�W
                                    Bxwd8  �          @�Q��`��@���@�A�  B��\�`��@�\)@EA�  C \                                    Bxwr�  �          @�p��]p�@��\?333@�  B����]p�@���?���A��RB��H                                    Bxw��  �          @�{�
=q@��\�(�����B�(��
=q@��\?z�@�Q�B�#�                                    Bxw�*  �          @��E�@�ff����ffB��)�E�@����z�H�33B���                                    Bxw��  �          @�G��{@�{>��R@9��Bި��{@�\)?ǮAk�
B��                                    Bxw�v  �          @��?�=q@����\)��(�B�?�=q@��Ϳ�=q�0��B�\)                                    Bxw�  �          @�G����
@�p�>���@G
=BΙ����
@�
=?��RAx  Bϔ{                                    Bxw��  �          @��
�%@��
������B���%@��?!G�@ǮB�                                    Bxw�h  �          @���W
=@�ff��ff�^�\B�k��W
=@�������B��H                                    Bxw�  �          @������@��ͿaG��z�B�=q����@�ff>�G�@��B�
=                                    Bxw��  �          @�Q�8Q�@��Ϳ���%�B��8Q�@��>k�@�
B�aH                                    BxwZ  T          @�p����H@�33=��
?B�\B��
���H@�ff?���AO�B�                                    Bxw   �          @�p�>��
@����Q��yG�B���>��
@�33����3�
B�=q                                    Bxw"�  �          @����\@�
=��(��g33B�\)��\@����\)�,(�B��
                                    Bxw1L  �          @�(����@�
=���H��=qB�\)���@��ÿ@  ��\B��                                    Bxw?�  �          @���{@����p����B����{@��
>��R@9��B�k�                                    BxwN�  �          @��   @��H�޸R��G�B�#��   @��H��\���B�#�                                    Bxw]>  �          @�p���@�z��z����HB���@��+��ǮBߣ�                                    Bxwk�  �          @���p�@�p��z�����B�aH��p�@���B�\���B��                                    Bxwz�  �          @�=q�=p�@��R�5����B��H�=p�@��R��z���(�B�W
                                    Bxw�0  �          @����ff@��H�R�\�G�B�����ff@�ff�Q���(�B�u�                                    Bxw��  �          @�33�n{@��H�c�
�G�Bɳ3�n{@�Q������G�B��                                    Bxw�|  �          @�  �@  @�{?���Al��B��@  @�(�@I��A�ffB�{                                    Bxw�"  �          @�ff�e@�  @W�A�p�B�B��e@��@�Q�B�RB���                                    Bxw��  T          @�  �n{@�  @w
=A���B���n{@�Q�@�B$ffC.                                    Bxw�n  T          @�z��[�@��@[�Aݙ�B�W
�[�@���@��\B��B�Ǯ                                    Bxw�  �          @�  �K�@�p�@O\)A�{B�\�K�@���@�p�B33B�                                    Bxw�  �          @�G��>�R@�@UA�Q�B�z��>�R@���@�
=B�B�p�                                    Bxw�`  �          @��ff@��?���A��HB�=q�ff@�{@C�
A�p�B�#�                                    Bxw  �          @��H�!G�@��H?�  An�\B����!G�@�G�@Dz�A�Q�B��                                    Bxw�  �          @��I��@�=q@1G�A��B��H�I��@���@�  B��B��                                    Bxw*R  �          @�z��o\)@�ff@\(�A�=qB����o\)@���@�(�B\)C \                                    Bxw8�  T          @��H�h��@�  @VffAхB���h��@�33@��B�\B�Ǯ                                    BxwG�  T          @�33�w�@�
=@fffA�{B��=�w�@���@��B�C�\                                    BxwVD  �          @���u@�\)@`��A�G�B�\�u@���@���B�C+�                                    Bxwd�  �          @�Q��n�R@�{@G
=A�Q�B�u��n�R@�33@��B��B�(�                                    Bxws�  �          @���j�H@�{@:�HA�G�B�B��j�H@���@}p�B33Cz�                                    Bxw�6  T          @��h��@�33@�A���B�� �h��@�@aG�A�p�B���                                    Bxw��  �          @�Q��z=q@�z�@,(�A��RB�� �z=q@���@q�BC�)                                    Bxw��  �          @�\)���
@���@)��A���C ���
@�G�@qG�A��HC@                                     Bxw�(  �          @�\��  @��H@J=qẠ�C���  @�  @�B
��C
^�                                    Bxw��  �          @�\��{@�
=@L(�A���C����{@hQ�@��
B��C�q                                    Bxw�t  �          @��p  @�@<��A��
C �3�p  @x��@|(�B\)C�3                                    Bxw�  �          @�{�xQ�@���@@��A�=qC���xQ�@n�R@}p�BQ�C+�                                    Bxw��  �          @׮�P��@�G�@VffA�  B�ff�P��@z=q@�33B"  C�\                                    Bxw�f  �          @����)��@qG�@�\)B6�B�33�)��@&ff@��RB_33C��                                    Bxw  �          @�{�W
=@��@$z�A��B�(��W
=@��
@uB\)B�                                    Bxw�  �          @�ff�ff@�  @8��AՅB�(��ff@�ff@�Q�B(�B�\                                    Bxw#X  �          @��
�1�@�ff@8Q�A�G�B�R�1�@��@{�B��B��=                                    Bxw1�  �          @���5�@�{@-p�A��
B�5�@�{@qG�BB�
=                                    Bxw@�  �          @أ��a�@�{@:=qA���B��a�@���@}p�B(�C^�                                    BxwOJ  �          @�\)���@���@N{A�
=CxR���@[�@��
B��CǮ                                    Bxw]�  �          @��H�y��@�@^{A�CW
�y��@aG�@�z�B�C	�                                    Bxwl�  �          @��H��Q�@]p�@�33B33C� ��Q�@�@���B,(�C��                                    Bxw{<  �          @�G����@�@~{B�\C����?��@�z�B(��C$Q�                                    Bxw��  �          @�\)���\?�=q@Z=qB
C33���\?�  @o\)Bp�C'��                                    Bxw��  �          @�z����R@{@hQ�B=qC+����R?�=q@�G�B)�HC"z�                                    Bxw�.  �          @�z��|(�@G�@�
=B.p�C�f�|(�?}p�@�=qBA\)C%޸                                    Bxw��  �          @�ff�^�R?У�@z�HB6ffC��^�R?.{@�{BG�C(�                                    Bxw�z  �          @ȣ�����?��H@|(�B��C%{����>�=q@��B(33C0�{                                    Bxw�   �          @�  ����@aG�@e�B�C
ٚ����@&ff@���B'p�C.                                    Bxw��  |          @�  �Vff@�z�@VffA��B����Vff@`  @���B%C�q                                    Bxw�l  �          @�ff�!G�@��?��A�(�B螸�!G�@���@@��A�
=B�=q                                    Bxw�  T          @�33�W
=@�=q�.{�	��B���W
=@�\)?c�
A0z�B�33                                    Bxw�  �          @����p�@��\?   @�B�  ��p�@�=q?�(�A��B�G�                                    Bxw^  �          @��R�@  @�p��\)��p�B�z�@  @����z�H�{B�\)                                    Bxw+  �          @��H����@��ÿ��
�W�B�B�����@�ff���
�O\)B��                                    Bxw9�  �          @�
=�I��@��H�!G���33B�3�I��@��H?0��@���B�                                    BxwHP  �          @�G��g
=@��>L��?�Q�B�W
�g
=@�G�?��
AP��B�8R                                    BxwV�  �          @�(��p��@��>�G�@hQ�B�p��p��@�\)?�G�AmB��H                                    Bxwe�  �          @�Q��tz�@��>��@p�B����tz�@�33?�ffAT��B�{                                    BxwtB  �          @����x��@���<��
>��B��=�x��@��?��
A1��B��                                    Bxw��  �          @����\(�@��׽�Q�Q�B��)�\(�@�z�?���A*�\B��                                    Bxw��  �          @�G��P��@��
�����p�B�33�P��@��\?\(�@�=qB�                                    Bxw�4  T          @���:=q@��׾�33�\��B�(��:=q@�
=?W
=A��B�q                                    Bxw��  �          @�33� ��@������z=qB�.� ��@�?c�
A\)B��                                    Bxw��  T          @����W
=@�(�?Tz�@�  B���W
=@���@�A��B�k�                                    Bxw�&  �          @�Q��Fff@���?aG�@�=qB�z��Fff@���@(�A��B���                                    Bxw��  �          @��
��H@��\?Tz�@��B�\��H@��R@�A���B�R                                    Bxw�r  �          @Ǯ�ff@��\?�A.ffB�q�ff@�z�@��A��B�(�                                    Bxw�  �          @Ϯ�L(�@�33?L��@�G�B�k��L(�@�  @ffA��HB��                                    Bxw�  �          @���"�\@�G�?+�@ə�B�=q�"�\@�
=?�(�A�B��
                                    Bxwd  �          @��
���@��?333@��HB�u����@��\@G�A���B���                                    Bxw$
  �          @������@���?s33AffB�z����@�(�@\)A�\)B�\)                                    Bxw2�  �          @���p�@��
?xQ�AB�.�p�@�\)@{A�z�B�                                     BxwAV  �          @�
=���H@�ff?��Aa��B�(����H@��@p�Aأ�BݸR                                    BxwO�  �          @�p����@���?���Ab�\B�  ���@�{@p�A��
B��f                                    Bxw^�  �          @�33��Q�@��?s33A��B�G���Q�@�@(�A�
=BԀ                                     BxwmH  �          @�\)�s33@�G�?fffA��B�k��s33@�p�@��A��B���                                    Bxw{�  
�          @�Q쿐��@��R?�\)A\  B�Q쿐��@�\)@(��Aڏ\B͊=                                    Bxw��  �          @�  �   @�  ?�
=Ar�\B��Ϳ   @�  @)��A��B��                                    Bxw�:  |          @�
=��@��?�A�{B��q��@��
@333B�RB��
                                    Bxw��  �          @��R=#�
@�p�>�33@w�B�{=#�
@�{?˅A�z�B�                                    Bxw��  �          @�{>�Q�@�z�?�@��B�W
>�Q�@�33?���A�Q�B��f                                    Bxw�,  �          @��=u@���>�ff@��HB�Ǯ=u@�Q�?��HA��\B��R                                    Bxw��  �          @�\)>�\)@�33?}p�A'�
B��>�\)@�ff@��A��
B�k�                                    Bxw�x  �          @�p�?�R@��H?(�@�B���?�R@���?��A��HB�                                      Bxw�  �          @�p����H@���?�@�Q�B�����H@�\)?�ffA�  B��                                    Bxw��  �          @����
=q@��\?�@���B����
=q@���?޸RA�z�B�W
                                    Bxwj  �          @�33�   @�ff?�@�(�B����   @�p�?��
A�  B�.                                    Bxw  �          @�=q�   @���?.{@�
=Bޅ�   @��\?�z�A���B�{                                    Bxw+�  �          @�=q��G�@��R?J=qA\)B�\)��G�@��@�\A��\B��)                                    Bxw:\  �          @�p���=q@���>��H@���BԳ3��=q@��?�ffA���B�k�                                    BxwI  �          @�p��+�@��H?aG�A�RB��+�@�\)?�p�A�ffB��3                                    BxwW�  �          @����\)@Vff?�Q�A@z�C&f��\)@=p�?�p�A��C��                                    BxwfN  �          @����ff@l(�>�@���C\)��ff@]p�?�\)A[33C.                                    Bxwt�  �          @��{�@�33>��@)��C�\�{�@z=q?�G�AMp�C.                                    Bxw��  �          @�
=�x��@��
?:�H@�G�CY��x��@tz�?�(�A�{C�\                                    Bxw�@  �          @�ff���@}p�?
=@��C�3���@l(�?�ffA}�C	�R                                    Bxw��  �          @�ff����@tz�?@  @��
C	c�����@`��?�
=A���C�                                     Bxw��  �          @�(��{�@|(�?E�@�33C���{�@hQ�?�(�A��\C	G�                                    Bxw�2  �          @�z��hQ�@�ff?xQ�A�RCٚ�hQ�@tz�?�p�A���C}q                                    Bxw��  �          @�{�`��@���?��AUp�Cc��`��@s�
@�A��
C��                                    Bxw�~  �          @����QG�@�?�=qAY��B�Ǯ�QG�@}p�@��A�  C��                                    Bxw�$  �          @����(��@�33?�p�Ax��B�q�(��@��H@%�A�33B�                                    Bxw��  �          @����H@�33?��AN=qB�3��H@��H@(��A�B��                                    Bxwp  �          @�����H@���?�Ao33B�p���H@�@Dz�A�B��                                    Bxw  �          @����@�\)?���Ag�B��f��@�z�@?\)A�ffB�k�                                    Bxw$�  �          @�{��z�@�p�?�ffAh  B�.��z�@�33@:�HA�B�                                      Bxw3b  �          @�녿�  @��
@\)A���B�B���  @�(�@aG�BQ�B��                                    BxwB  �          @��H���H@��@p�A��B�.���H@��@p  B  B�{                                    BxwP�  �          @�33���@Ǯ?��
A\)B�z���@���@$z�A�\)B���                                    Bxw_T  �          @��ÿ�p�@��?Tz�@��B׳3��p�@�  @
=A�\)B�\                                    Bxwm�  �          @�  ��ff@�p�?xQ�A	G�BԀ ��ff@�
=@   A�
=B��H                                    Bxw|�  �          @�Q��Q�@�p�?.{@��B�녿�Q�@���@{A�B�
=                                    Bxw�F  �          @У׿�33@�ff?!G�@���B�  ��33@��H@�A��\B���                                    Bxw��  
�          @�Q쿝p�@�33?333@ƸRB��f��p�@��R@33A�\)B�L�                                    Bxw��  �          @�G����@�(�?��@���B�=q���@�Q�@p�A�{B̙�                                    Bxw�8  �          @�33��=q@�p�?O\)@���B�zῪ=q@�  @�HA�  B�\                                    Bxw��  �          @˅��G�@�33?G�@ᙚB��Ϳ�G�@�ff@�
A�\)BѮ                                    BxwԄ  �          @Ǯ��z�@���?E�@�\B�LͿ�z�@��
@�\A�(�B�{                                    Bxw�*  �          @ə���Q�@���?�{A"ffB��)��Q�@���@'�A���B��                                    Bxw��  �          @�\)��{@���?��AjffBȀ ��{@���@J=qA�ffB�Ǯ                                    Bxw	 v  T          @���Q�@��@�RA�G�B�{��Q�@�G�@mp�B#ffB�z�                                    Bxw	  �          @��H�B�\@�Q�?У�A��RB�ff�B�\@��@;�A�
=B�z�                                    Bxw	�  �          @��\�#�
@��
@
�HA��B�.�#�
@�z�@Z�HB��B�33                                    Bxw	,h  �          @��
?Tz�@��R@:�HA��HB�?Tz�@s33@�G�B8ffB�Q�                                    Bxw	;  
�          @����p�@�{@   A���B��\��p�@�  @Mp�B��B��                                    Bxw	I�  �          @�{>�Q�@�z�?�Q�A���B��>�Q�@���@>{B��B���                                    Bxw	XZ  �          @����\@��\?�\)A��\B��H��\@�
=@<��A��RB�B�                                    Bxw	g   �          @��H����@���?���A[
=B�zῌ��@���@*=qA��
B�                                      Bxw	u�  T          @�
=>Ǯ@�G�?��HA�\)B��>Ǯ@��@Mp�B�B��R                                    Bxw	�L  �          @��þ8Q�@��R?�G�A�33B�
=�8Q�@��@C�
BQ�B���                                    Bxw	��  �          @��\��G�@��H@33A��\Bʅ��G�@��
@S�
B��B��                                    Bxw	��  �          @�33���@���@
=A��B�G����@�p�@X��B�B�Q�                                    Bxw	�>  �          @����  @�33@33A�B�����  @��@c�
B�HB���                                    Bxw	��  �          @�=q�(�@���@�RA�33B��H�(�@���@^�RBG�B�Q�                                    Bxw	͊  �          @��H��@��\@G�A�G�B��f��@���@a�B��B�                                    Bxw	�0  �          @�(��!G�@�G�@   A���B��{�!G�@�{@s�
B!ffB�(�                                    Bxw	��  T          @�(����@�33@�A�{B�Ǯ���@���@l��B(�B�{                                    Bxw	�|  T          @�33��@�z�@J=qB�
B�\)��@xQ�@�33B@{BÏ\                                    Bxw
"  T          @��ͿO\)@�=q@9��A��B�B��O\)@��@���B2��B�k�                                    Bxw
�  �          @��
�fff@��@.�RA��B�  �fff@�ff@\)B+B�8R                                    Bxw
%n  �          @�����@�G�@
=Aď\B̸R����@�\)@g�B��B�L�                                    Bxw
4  T          @�
=��@�p�?��HA��B�녿�@�G�@=p�B�RB�aH                                    Bxw
B�  �          @�  ��{@{����33C�=��{@4z�k����CxR                                    Bxw
Q`  �          @��R���������_\)���C7�f����?�R�\���  C,�
                                    Bxw
`  �          @�Q���
=���
�e��\)C@E��
=���
�n�R�G�C4E                                    Bxw
n�  �          @�Q�������
�tz��=qCGG������G����H�,  C9�q                                    Bxw
}R  �          @��H���R?�  �Vff�{C)!H���R?�\)�<����RC E                                    Bxw
��  �          @У���{?u�tz���C)z���{?��H�Z�H���HCJ=                                    Bxw
��  �          @ҏ\���H?�������\)C(xR���H@�fff��
C�H                                    Bxw
�D  �          @љ����R?��
�o\)��C&:����R@  �P  ���HC��                                    Bxw
��  �          @������H?�z��]p�� ffC%8R���H@33�<�����C�3                                    Bxw
Ɛ  T          @ҏ\��G�?�Q��L����\)C%k���G�@���+���
=C�\                                    Bxw
�6  �          @�33���?�
=�B�\��\)C#G����@���p���33Cc�                                    Bxw
��  �          @��H����@(��2�\��Q�Cu�����@7�����G�C�f                                    Bxw
�  �          @����z�@{�2�\��p�Cff��z�@H���   ��ffC�=                                    Bxw(  �          @�Q���  @*=q�1G��ə�C#���  @Tz��
=��(�C��                                    Bxw�  �          @�p����R@*=q�B�\��
=C�q���R@Y���(���  C��                                    Bxwt  �          @��H��\)?�Q��^�R�
=C!:���\)@%�8Q���G�C�=                                    Bxw-  �          @�ff��z�?����c�
�
=C&:���z�@���E��z�Cc�                                    Bxw;�  �          @�p���=q?h���p  ���C)E��=q?��Vff���CG�                                    BxwJf  �          @���Q�?����X����HC%5���Q�@p��8���ޣ�C(�                                    BxwY  �          @�33��
=?}p��>{���
C)B���
=?��
�%��ȏ\C!.                                    Bxwg�  �          @��
���?����1G���RC'^����?�{����
C�\                                    BxwvX  �          @�(���z�?�(��=p���
=C"J=��z�@�R����CJ=                                    Bxw��  �          @�����z�?޸R�aG���C c���z�@*=q�8���ۅCxR                                    Bxw��  �          @�{�|(�@!��*�H��C:��|(�@K��������HC�                                    Bxw�J  �          @���W�@(����R��G�C��W�@N�R�����33C33                                    Bxw��  �          @��\�g�@Tz��&ff��p�C	u��g�@z�H�Ǯ����C��                                    Bxw��  �          @�  �j�H@J=q�AG����
CG��j�H@xQ�� ������CaH                                    Bxw�<  �          @�����p�@%��H����C33��p�@W���\����C\                                    Bxw��  �          @�����
=@U>�@��C{��
=@Dz�?�z�AXz�CL�                                    Bxw�  �          @�����@aG����H�S�C�����@r�\����hQ�C��                                    Bxw�.  �          @����33@G������%�C� ��33@S�
�B�\��G�C=q                                    Bxw�  �          @�{����@@�׿����(��C8R����@Mp��aG��C�H                                    Bxwz  �          @�(����\@G���\)�0��C)���\@S�
�B�\��C�=                                    Bxw&   �          @�����=q@?\)�����{�C0���=q@Tz�+��ҏ\Ck�                                    Bxw4�  �          @������@0  �����Xz�C�R���@AG���\����CY�                                    BxwCl  T          @�������@'��У���
=C������@>�R�O\)� ��C�                                     BxwR  �          @�{��{@G������
C�q��{@HQ�>���@tz�C�)                                    Bxw`�  �          @ʏ\��
=@dz�?�=qAg�CQ���
=@=p�@"�\A�C8R                                    Bxwo^  �          @�ff���@j�H@��A�
=C�����@4z�@VffB{C��                                    Bxw~  �          @�Q���
=@7�����\)C����
=@S33��G��ffC�q                                    Bxw��  �          @�p���G�@*�H��{�J�HCL���G�@)��>�ff@�p�Cn                                    Bxw�P  �          @�\)��ff@B�\������
C�f��ff@C33>�G�@x��C޸                                    Bxw��  T          @���(�@6ff�:�H��(�C�q��(�@<(�>�?�p�C:�                                    Bxw��  �          @�33��p�@\)�s33���C0���p�@*=q�8Q���C��                                    Bxw�B  �          @������
@ �׿����)��C�����
@-p���z��5�C)                                    Bxw��  �          @�z���\)@>�R�.{����C#���\)@B�\>k�@{C��                                    Bxw�  �          @��R��
=@(Q�@  ��\C���
=@/\)=u?
=C0�                                    Bxw�4  �          @�{��{@.{=���?�ffC&f��{@$z�?fffA
=C��                                    Bxw�  T          @�\)��z�@>�R=L��>�G�C���z�@5�?k�A
{CO\                                    Bxw�  �          @�����@W���Q�:�HCG����@P  ?h��@�
=C33                                    Bxw&  �          @���\)@Z=q�����Q�C8R��\)@Y��?�@�33CT{                                    Bxw-�  �          @�{���@HQ�!G����C޸���@K�>���@4z�C��                                    Bxw<r  �          @�  ���@i����Q��K�C�R���@e?=p�@�=qC#�                                    BxwK  �          @����G�@g��k��C����G�@a�?\(�@��CT{                                    BxwY�  �          @θR��33@e������a�C:���33@b�\?0��@�z�C�\                                    Bxwhd  �          @���=q@G��������C����=q@I��>�Q�@L(�C�                                     Bxww
  �          @�p���z�@Z=q�����C����z�@Z=q?�@�z�C�3                                    Bxw��  �          @����@Dz�z���p�CO\���@Fff>�p�@S�
C\                                    Bxw�V  �          @�p���  @N{�&ff���C����  @QG�>�33@I��CG�                                    Bxw��  �          @�(����\@@  �!G���
=C� ���\@C33>���@,��CW
                                    Bxw��  �          @�G���
=@Dzᾮ{�EC����
=@A�?(�@���C                                    Bxw�H  �          @�
=����@8Q�L�;�CxR����@0��?Q�@�Cs3                                    Bxw��  �          @�p����@N{��\)�(��Ck����@J=q?5@ӅC�3                                    Bxwݔ  �          @����R@h�ÿ�����
C�����R@i��?�@�C�H                                    Bxw�:  �          @�{��  @J�H��33�Q�C޸��  @HQ�?!G�@���C8R                                    Bxw��  �          @�G�����@g
=��\)�!G�C������@]p�?��A�
C�                                    Bxw	�  
�          @�ff����@Mp����Ϳh��C�����@E?c�
Az�C��                                    Bxw,  �          @�(���G�@'��#�
��Q�C����G�@   ?E�@�{C�                                    Bxw&�  �          @�\)��G�@R�\��\)�!G�C���G�@J=q?s33A�
C0�                                    Bxw5x  �          @�
=���@C�
�#�
��G�C�����@=p�?J=q@�Q�CQ�                                    BxwD  �          @Ǯ��ff@@  >��@p��C0���ff@.{?��AB�RCp�                                    BxwR�  T          @����z�@P  >.{?��
C�f��z�@B�\?�A*�RC�{                                    Bxwaj  �          @ȣ���(�@e�����z�C#���(�@\(�?}p�AG�C.                                    Bxwp  �          @ə���
=@u���ff��33C^���
=@q�?B�\@�ffC��                                    Bxw~�  �          @�����@1녿(����Q�C\)���@6ff>u@�C��                                    Bxw�\  �          @�������?��Ϳ�
=�w�
C'�����?����,  C"�                                    Bxw�  �          @�(����R?�׿������C .���R@�ÿ�G��G
=CW
                                    Bxw��  T          @�  ��
=?��Ϳ�z���ffC!T{��
=@녿�  ��Ch�                                    Bxw�N  �          @�{���?�녿�
=�|Q�C!� ���@���G��p�C�f                                    Bxw��  �          @�33��\)@���{����C �\��\)@#�
��{�
=C��                                    Bxw֚  �          @Ӆ����@33��\�x��C!0�����@ �׿��
�  Ck�                                    Bxw�@  �          @��
���
?��������L(�C �H���
@G��(���ǮC��                                    Bxw��  �          @�������?�p�����H��C W
����@�\��R��(�C�                                    Bxw�  �          @\��ff?�
=��p�����C ���ff@Q쿃�
��Cn                                    Bxw2  T          @�Q����?ٙ��$z���Q�C!�����@��������CǮ                                    Bxw�  T          @�(����R?���+��υC!����R@#33��(���=qC�3                                    Bxw.~  �          @Å���?����2�\��C ^����@'���
��(�C�H                                    Bxw=$  �          @�����H?���-p���=qC#���H@Q������G�C{                                    BxwK�  �          @�(����H?�  �9����(�C!�
���H@ff�G���  C�{                                    BxwZp  �          @\���
?�p��U�(�C#\���
@{�,(���C
=                                    Bxwi  �          @����z�?���U��{C$�R��z�@�.{��G�Cz�                                    Bxww�  �          @������R?˅�R�\�
=C!aH���R@#�
�'
=��33Cs3                                    Bxw�b  �          @��R��(�?ٙ��>{��C �\��(�@$z��G����HC=q                                    Bxw�  �          @�z���p�@  �'�����Ck���p�@>�R��G���C��                                    Bxw��  �          @�
=��\)@4z��\)���C33��\)@Y�����H�;33CW
                                    Bxw�T  �          @�����G�@C33��z��2{C�{��G�@P�׽�G���G�C)                                    Bxw��  �          @�����z�@A녿fff�	p�C�=��z�@I��>�?�  Cz�                                    BxwϠ  �          @�{��\)@<(���
=�8z�C� ��\)@J=q�#�
���C��                                    Bxw�F  �          @�p���{@<�Ϳ�G��EC!H��{@L�;k��p�C
=                                    Bxw��  �          @�ff���\@L�Ϳ�\)�U�Cs3���\@^{��  �Q�CB�                                    Bxw��  �          @�Q���p�@H�ÿ��
�E�Ck���p�@X�þ8Q���HCn                                    Bxw
8  �          @�{��\)@7
=��  �i��C#���\)@L(������=qCW
                                    Bxw�  �          @�ff����@&ff�����aG�C@ ����@;���\��Cc�                                    Bxw'�  �          @�G����@33�����]G�C�q���@(Q�
=q��z�C�                                    Bxw6*  �          @��H���\@��\)��{C�����\@%�����)�C��                                    BxwD�  �          @��dz�@\�������  C���dz�@�����\)�;
=Cff                                    BxwSv  �          @�����@�  ��=q���
B鞸���@��H����\B�                                    Bxwb  �          @�����H@�z��z���=qB�����H@���8Q����B�\                                    Bxwp�  �          @����   @�(��=q����B����   @�p��:�H��  B�                                    Bxwh  �          @�ff�$z�@��\�p���=qB�{�$z�@�녿������B��f                                    Bxw�  �          @�  �33@�  ����
=B�R�33@�\)�\)��33B�p�                                    Bxw��  �          @����R@�=q���H��
=B�k���R@�ff�k���\B���                                    Bxw�Z  �          @��33@�G��\)�иRB����33@���Q����B�                                    Bxw�   �          @����@�33�Q���=qB��@�(��333����B�{                                    BxwȦ  �          @�ff��@�녿޸R��Q�B��)��@��H>��?��HB��                                    Bxw�L  �          @�Q���H@��ÿ˅�u��B�ff���H@��>�p�@dz�B�33                                    Bxw��  �          @Å�޸R@��R����%G�B�녿޸R@��?h��A	�Bսq                                    Bxw��  �          @����33@�(������p��B�8R��33@��H>��@tz�B�                                      Bxw>  �          @�  ��@�=q����F{Bօ��@�?8Q�@�z�B��                                    Bxw�  �          @�\)�Q�@�
=��{�$��B�Ǯ�Q�@�Q�?h��A=qB܊=                                    Bxw �  �          @ƸR���@�녿n{�33B��ÿ��@���?�{A$��B�#�                                    Bxw/0  �          @�z���@��>�  @
=B�q��@�{@	��A��B�
=                                    Bxw=�  �          @�
=���@����������B�z���@��=��
?O\)B�                                    BxwL|  T          @�Q쿨��@��*=q�ܣ�B������@�G��O\)�p�B��H                                    Bxw["  �          @��ÿ��@���>�R���B�{���@�녿�p��EG�B�=q                                    Bxwi�  �          @��H���H@}p��XQ���B����H@�z����p�B�Ǯ                                    Bxwxn  �          @���\@�������B�(��\@����\)�;�BӔ{                                    Bxw�  �          @��R��p�@���
���
Bڔ{��p�@��H�aG����B��
                                    Bxw��  �          @�(���33@�z��G����
B��f��33@�{?��At��B��)                                    Bxw�`  �          @�=q��
=@�G��Tz���B��)��
=@��?�33A7�B�(�                                    Bxw�  �          @�p����R@����G���33Bҏ\���R@�녿
=q��p�Bϣ�                                    Bxw��  �          @�\)�#�
@�p���\��\)B��=�#�
@�zᾳ33�`��B�=q                                    Bxw�R  �          @�=q��G�@�p��z���p�B�B���G�@����L�;��B��                                    Bxw��  �          @�33>\)@�����%B�B�>\)@�{?��\A"{B�B�                                    Bxw�  �          @�p���{@�33�<����\B�aH��{@��\����&�HB�u�                                    Bxw�D  �          @�����
@�
=��z���\)B�� ���
@�{>�@�p�B�u�                                    Bxw
�  �          @��>8Q�@�ff������\)B�aH>8Q�@���>8Q�?ٙ�B��\                                    Bxw�  �          @�p�?Y��@�\)��  ���B��
?Y��@�{?��A��\B�                                      Bxw(6  �          @�33@p�@��=u?�B�=q@p�@�@G�A�
=B�.                                    Bxw6�  �          @���@G�@�z�>��@(�B�\@G�@�@\)A��RB���                                    BxwE�  �          @Å@�@�  ?�\)AV�RB�  @�@�p�@H��B
=Bz�
                                    BxwT(  �          @�p�@�R@���?��
A=qB��@�R@��@>{A��HB�(�                                    Bxwb�  �          @�=q@
=@���>�p�@c�
B��@
=@�z�@
=A�(�B�L�                                    Bxwqt  T          @�z�?��@��H=�?��HB��?��@�p�@
=A���B��H                                    Bxw�  �          @��H@=q@�{?h��A�B�(�@=q@�  @333A݅B|\)                                    Bxw��  �          @���?�{@��R=L��>�B��H?�{@�=q@ffA�33B�aH                                    Bxw�f  �          @�\)?��
@�p����
�E�B��{?��
@��H?�\)A�p�B�\)                                    Bxw�  �          @�=q>��@�  �8Q���RB��=>��@�z�?��AY��B�k�                                    Bxw��  �          @�z�>�
=@��׿���)�B�G�>�
=@���?��\A'33B�G�                                    Bxw�X  �          @�33>�  @�ff��
=�B�\B��3>�  @�Q�?^�RA�B��q                                    Bxw��  �          @��R��33@�{�'��ϮB�{��33@�G��\)��z�B��                                    Bxw�  �          @�녿�G�@�ff����33B�{��G�@��R��z��-p�B�Q�                                    Bxw�J  �          @���xQ�@��\�0  ��=qBɅ�xQ�@�\)�8Q��ᙚB��                                    Bxw�  �          @��
���H@�\)�'
=��G�B��;��H@�=q�����B��3                                    Bxw�  �          @��H�#�
@����=q��33B��H�#�
@��þ��
�J=qB���                                    Bxw!<  �          @�{>�  @��R�B�\�33B���>�  @����{�6ffB��3                                    Bxw/�  �          @�?�R@�
=�ff��{B��=?�R@�(��#�
��(�B��                                    Bxw>�  �          @�=q?k�@�p���G���Q�B��)?k�@��R>��@*=qB��f                                    BxwM.  �          @�Q�=�Q�@�ff��  ��33B�\=�Q�@�\)>�\)@8��B�(�                                    Bxw[�  �          @�ff�W
=@�33��{��ffB���W
=@�>\)?��
B�ff                                    Bxwjz  �          @�  ���@��L�����B������@�������P��B�(�                                    Bxwy   �          @�Q��6ff@���Q����B��R�6ff@�z῱��M�B�3                                    Bxw��  �          @�33�333@�Q��^{��B�\�333@�\)��ff�b=qB��                                    Bxw�l  �          @�ff�Vff@��QG���B�B��Vff@��\����G�B�ff                                    Bxw�  �          @����\(�@��\�`����C&f�\(�@��H��  �~=qB�{                                    Bxw��  �          @љ��`  @XQ���33�'�C��`  @�Q��1���\)B���                                    Bxw�^  �          @�{�o\)@b�\�w��C�
�o\)@�
=�G���G�C \)                                    Bxw�  �          @���n�R@^�R�k��33C	  �n�R@�33�
=��
=C
=                                    Bxwߪ  S          @Ǯ�z=q@fff�7
=��=qC	T{�z=q@����ff�G�C�=                                    Bxw�P  �          @�Q��|(�@��:�H�׮CG��|(�@�
=����#33C aH                                    Bxw��  �          @�33����@c�
��(��G�CaH����@�33�\)����C��                                    Bxw�  �          @޸R��G�@Fff��Q��!C!H��G�@�=q�@����\)C.                                    BxwB  �          @޸R����@=p���z��'  C=q����@���K���G�C�=                                    Bxw(�  �          @�{���\@H�������(�Cn���\@���G
=��\)C5�                                    Bxw7�  �          @����r�\@dz���
=�(�HC�q�r�\@��H�AG���B�aH                                    BxwF4  �          @�\�e�@\����  �4C��e�@��H�Tz����HB�8R                                    BxwT�  �          @�(��s33@HQ������9(�C}q�s33@�33�dz�����C �                                    Bxwc�  �          @��
�p  @H�����\�2�C�p  @���Q��㙚C L�                                    Bxwr&  �          @�{���@XQ�����!z�C:����@��H�7���G�C                                    Bxw��  �          @�z����\@^�R��\)�z�C5����\@��3�
����CO\                                    Bxw�r  �          @������@Vff��33� p�C&f���@���>{�ģ�C�)                                    Bxw�  �          @�p����@J�H����%33C����@�  �J=q����C��                                    Bxw��  �          @�(���z�@:=q����)�C}q��z�@����U���z�C�                                    Bxw�d  �          @�(����@*�H����.\)C�)���@�33�`����{Cn                                    Bxw�
  �          @�p���33@2�\��Q��)(�C����33@���E��p�C��                                    Bxwذ  �          @Ϯ���?�(���{�9��CO\���@X���j�H�
{Cp�                                    Bxw�V  T          @���O\)@dz��P���(�C5��O\)@�����\)���B��)                                    Bxw��  �          @�(��,(�@�(���\)��Q�B��)�,(�@�
=>#�
?�{BꞸ                                    Bxw�  �          @�=q�"�\@���\)���B�p��"�\@���k����B�Ǯ                                    BxwH  �          @��ÿ�z�@�Q��:=q��\)B�=q��z�@�녿��\�+�
B�\)                                    Bxw!�  �          @��H�
�H@��\�/\)���
B���
�H@�녿W
=�
ffB�\)                                    Bxw0�  �          @����=q@�33�	����  B�LͿ�=q@�����Q�aG�B�                                    Bxw?:  �          @�(���z�@���33��Q�B�zῴz�@�p�?�\@�{B�=q                                    BxwM�  �          @�����@�33�˅�{\)B�\���@�G�?!G�@�{B��
                                    Bxw\�  �          @����@�z��ff�p��B�G���@��?.{@���B�
=                                    Bxwk,  �          @����"�\@�ff��
=�8��B���"�\@�  ?s33AQ�B�3                                    Bxwy�  �          @��H��@��\���
�s�Bـ ��@�  ?0��@�Q�B�ff                                    Bxw�x  �          @�Q����@��Ϳ����@��B�ff����@�{?��\A$  B�33                                    Bxw�  T          @�33��33@��׿h����HB�G���33@�?��AUp�B���                                    Bxw��  �          @�33����@���J=q����B�LͿ���@��?�Q�Ad(�B�(�                                    Bxw�j  �          @��\��z�@��׿5��  BՊ=��z�@�33?��Au�B֊=                                    Bxw�  �          @�ff�˅@�{�
=��ffB�(��˅@��R?��HA�=qB�k�                                    BxwѶ  �          @�Q��=q@�  ��ff��G�B�z��=q@�
=?:�H@У�B�z�                                    Bxw�\  �          @��
��{@�G�>���@>�RB�33��{@��@,��A�ffB��f                                    Bxw�  �          @�ff��{@�z�>���@fffBսq��{@�
=@3�
A��Bٔ{                                    Bxw��  �          @�{�\@ƸR>�=q@��B�z�\@��\@.�RA�  B�k�                                    BxwN  �          @�ff�3�
@��
�E���  B���3�
@��R?��AaG�B�u�                                    Bxw�  �          @�{��{@љ�>���@8Q�B��)��{@��
@:�HAӮB��=                                    Bxw)�  �          @�{?�ff@�ff?���A~�\B��H?�ff@�ff@��B��B�(�                                    Bxw8@  �          @��>�ff@��?�\)Af=qB��H>�ff@�(�@���B
=B�                                      BxwF�  �          @�녿�@�
=?(��@���B�aH��@��@Mp�A�\)B��R                                    BxwU�  T          @�\)��=q@�{>���@=p�B��)��=q@�Q�@9��AծB�p�                                    Bxwd2  �          @�
=�\@�Q�����c
=B�LͿ\@�(�?}p�A
=B�                                    Bxwr�  T          @�ff�4z�@�33�O\)�=qB�\�4z�@�녿���K
=B�G�                                    Bxw�~  �          @����^{@G
=�s33��
C
��^{@�z��{���HC W
                                    Bxw�$  �          @�33�P  @��R�	����33B�ff�P  @��=L��>�B���                                    Bxw��  �          @Ӆ�(�@Å��=q�Q�B߅�(�@�{@�RA���B�aH                                    Bxw�p  �          @У��+�@��   ��(�B���+�@��?��HA��B�                                      Bxw�  �          @�(���z�@(��|(��ffC�H��z�@u�&ff��Q�C
�{                                    Bxwʼ  �          @�{��\)@�R�z=q�\)C����\)@w
=�#33���RC5�                                    Bxw�b  �          @�{���@6ff�`  ���CE���@�G�����G�C��                                    Bxw�  �          @����k�@w
=�p���z�C�f�k�@��ÿ+��У�C
                                    Bxw��  �          @��H�`  @�Q�������C��`  @��H������B��R                                    BxwT  �          @�  �k�@�p��0  ����Cp��k�@��L����z�B�u�                                    Bxw�  �          @�  �j�H@�p���\��p�CO\�j�H@�  ��p��c33B�\)                                    Bxw"�  T          @��R�c33@��Ϳ�=q���C �H�c33@���>��?�z�B�\)                                    Bxw1F  �          @��H�hQ�@�ff��(��^�HB�L��hQ�@�z�?��@�(�B�#�                                    Bxw?�  �          @\��{@�녿���{33C����{@��
>B�\?�G�C                                    BxwN�  �          @�����  @������X��CY���  @���>Ǯ@mp�C��                                    Bxw]8  �          @�Q����@����z��Q�C.���@��?�@�=qC�3                                    Bxwk�  �          @Ϯ��  @�
=�!�����C�)��  @����ff�z�HC 
=                                    Bxwz�  �          @���|��@�Q�����
=C��|��@��R=u?   B�W
                                    Bxw�*  �          @��H�i��@����p���  B�{�i��@�
=>���@7�B�k�                                    Bxw��  �          @�(��_\)@��� ����{B�G��_\)@�33>�Q�@EB�Ǯ                                    Bxw�v  �          @��H�x��@�=q� ����p�Cn�x��@�ff��p��O\)B���                                    Bxw�  �          @�ff��ff@����
=���RC	����ff@�Q����ffC�H                                    Bxw��  �          @��
��z�@�ff�ٙ��f{C ���z�@�{?!G�@���B��{                                    Bxw�h  T          @�(���
=@�\)?p��@�B�(���
=@��R@k�B��B�p�                                    Bxw�  �          @ۅ�&ff@���?.{@�p�B�  �&ff@��
@]p�A�G�B��                                    Bxw�  �          @��
��=q@ָR>\)?�33B�G���=q@���@<��A��
B�L�                                    Bxw�Z  �          @�Q��R@ָR<�>k�B�z��R@�=q@7
=Aȏ\B��{                                    Bxw   �          @�Q�=p�@��
�s33�=qB�Ǯ�=p�@���?�
=A���B�8R                                    Bxw�  �          @�ff��z�@���˅�^{B�#׾�z�@�\)?��
A2{B��                                    Bxw*L  �          @�\)�,��@��
>�(�@l(�B�3�,��@��@>�RA�\)B�\                                    Bxw8�  �          @׮�#�
@�\)����\)B����#�
@��@'
=A��RB䞸                                    BxwG�  �          @�
=�Z�H@�G���\)�B���Z�H@��@p�A��B�{                                    BxwV>  �          @���'�@�33=��
?:�HB�k��'�@�\)@*�HA�  B�                                      Bxwd�  �          @��
��
@���>�\)@p�B�Q���
@��@<(�Aҏ\Bܨ�                                    Bxws�  �          @�ff>���@��H?s33A�B�L�>���@���@j�HB�
B�=q                                    Bxw�0  �          @�z῁G�@�Q�=�Q�?Q�BŨ���G�@�33@7�A�ffBǣ�                                    Bxw��  T          @��ÿٙ�@�\)�#�
��p�BҀ �ٙ�@���@A�(�B�{                                    Bxw�|  �          @�33�p�@��
�p����B�Ǯ�p�@�?�\Az�\B��                                    Bxw�"  �          @���@��ÿc�
�   B�=q��@��\?��
A��RB�p�                                    Bxw��  �          @�=q�33@����G��G�B����33@��?޸RAup�B�                                    Bxw�n  �          @��H��(�@ə��\�U�BҔ{��(�@�33@�A��B���                                    Bxw�  �          @ƸR���
@��
���B�uÿ��
@�
=?
=q@�Q�Bнq                                    Bxw�  �          @�{��@��\��(���Q�B�(���@Å?@  @�B���                                    Bxw�`  �          @ʏ\�L��@���G���z�B�\�L��@��?   @��B���                                    Bxw  �          @��
�#�
@�{��\��ffB��q�#�
@��H?�\@��B��3                                    Bxw�  �          @�
=>W
=@�ff��ff��33B��>W
=@˅?���Az�B�8R                                    Bxw#R  �          @ҏ\?^�R@�(�����4��B�z�?^�R@��?�{Ac\)B�L�                                    Bxw1�  �          @��
?�G�@�zΐ33��B�?�G�@�  ?�  At��B�G�                                    Bxw@�  �          @�(�?
=q@����
=�O�
B��f?
=q@��?�33AK�B��                                    BxwOD  �          @��>�Q�@�(����PQ�B�L�>�Q�@�(�?��AL��B�L�                                    Bxw]�  �          @ə�?(��@�(���
=�-��B�?(��@���?�{An=qB��\                                    Bxwl�  �          @ȣ�=�Q�@�=q�\�a�B�B�=�Q�@�(�?��A=B�G�                                    Bxw{6  �          @�����@�z��=q����B��
���@�=q?��Ap�B�                                    Bxw��  �          @�(�?�G�@���J=q��ffB�Ǯ?�G�@�(�?��RA�(�B��R                                    Bxw��  �          @�33?=p�@�\)�p���(�B�aH?=p�@�Q�?��A���B��H                                    Bxw�(  �          @��?
=q@�녿n{�{B��?
=q@�=q?�
=A���B�                                    Bxw��  �          @Ӆ?xQ�@���=�G�?n{B�#�?xQ�@�=q@>{A��B��                                    Bxw�t  �          @���?J=q@�\)�Y������B�\?J=q@�{@ffA�z�B�k�                                    Bxw�  �          @�p�?�@�
=��  �O�B��{?�@�
=?\AR=qB��{                                    Bxw��  �          @�(�?&ff@����  �Q�B�Q�?&ff@��?�p�AO�B�W
                                    Bxw�f  �          @Ӆ?0��@�Q���R��=qB�W
?0��@Ϯ?}p�A	p�B���                                    Bxw�  �          @�33?G�@ȣ׿����z�B��)?G�@θR?���A��B�B�                                    Bxw�  �          @��H��@�\)���H���B�G���@θR?�G�A��B�33                                    BxwX  �          @�(�� ��@���@����p�B��H� ��@��þ�p��R�\B�Ǯ                                    Bxw*�  �          @�  ��
=@�p��:�H��
=B����
=@����R�8��B��                                    Bxw9�  �          @�
=���@�
=�B�\��z�B����@�����\��B�k�                                    BxwHJ  �          @����@�G��B�\�뙚B����@�������(�Bמ�                                    BxwV�  �          @���
=@�G��5���B�LͿ�
=@�  �8Q��33B��
                                    Bxwe�  �          @�
=�ٙ�@�(��[��ffB�aH�ٙ�@��Ϳc�
��
B��                                    Bxwt<  �          @�(���@���
�H���B����@���>�@�=qB�B�                                    Bxw��  �          @�Q�>�ff@�녿�Q��@��B��R>�ff@�Q�?�
=Ah(�B���                                    Bxw��  �          @�\)>�
=@��Ϳ�33��  B��=>�
=@�G�?��A%��B��q                                    Bxw�.  �          @��R?0��@��R��  �jffB�=q?0��@���?�(�A=�B�ff                                    Bxw��  �          @��Tz�@�G������B�\�Tz�@�Q�?aG�A	G�B�k�                                    Bxw�z  �          @�z��G�@q��
=��=qC�=�G�@�{�����=qB�33                                    Bxw�   �          @����p�?У׿�z����C!�f��p�@z�xQ��!��C                                    Bxw��  �          @�
=��
=@n{�A����B�#׿�
=@�{���\�<  B�                                    Bxw�l  �          @�ff?5@���s33�)(�B�z�?5@�(����
�{
=B�
=                                    Bxw�  �          @��R?�(�@z�����l
=BU��?�(�@�  �O\)���B��                                    Bxw�  �          @��xQ�@p���\)�{�B���xQ�@���N{��Bή                                    Bxw^  �          @�  �]p�@=q�x���*C0��]p�@z=q�Q��ÅC�                                    Bxw$  �          @����c33@G��z�H�+�CaH�c33@s�
�p�����C��                                    Bxw2�  �          @����x��?��u��&�C�R�x��@W
=�%��ՅC0�                                    BxwAP  �          @��H���\?L���W����C*)���\@  �(Q���ffC�f                                    BxwO�  �          @���Dz�@,�����
�4��C
�f�Dz�@�������=qB�G�                                    Bxw^�  "          @�Q��|��?��R�w��*  CT{�|��@E�0�����C�3                                    BxwmB  "          @�{��\)?���Y����
C&33��\)@�R�#33�ۮC�f                                    Bxw{�  �          @����  @E�������HC  ��  @e�k��ffC�\                                    Bxw��  �          @����vff@~�R���
�vffC\�vff@��>��@��CE                                    Bxw�4  �          @��H���@g����R����C
�3���@�(���G���ffC0�                                    Bxw��  �          @�{���@���G����C�����@J�H�k��C��                                    Bxw��  �          @�ff��=q?�\����=qC.J=��=q?��ÿ�
=���C"��                                    Bxw�&  �          @�G�����@�#33���HC�=����@A녿�ff�ZffCk�                                    Bxw��  �          @�  �]p�@�=q��\���B�\)�]p�@��
?�@�=qB��
                                    Bxw�r  �          @Å�vff@�����33C#��vff@�{?��@�Cn                                    Bxw�  �          @\�k�@�G���{�v�\C�k�@���?+�@ʏ\B�G�                                    Bxw��  �          @ȣ��Fff@��   ���\B�ff�Fff@���@z�A�  B��                                    Bxwd  �          @ə��Tz�@����+����
B��Tz�@�Q�?�{A��B���                                    Bxw
  �          @�=q�[�@�G����R�5�B���[�@��\@
�HA�G�B�                                    Bxw+�  �          @�{�Z=q@����\)�&ffB���Z=q@�ff@	��A�(�B��f                                    Bxw:V  �          @�p��W�@�z��\����B�� �W�@�G�?�
=A�{B�=q                                    BxwH�  �          @�ff�i��@�
=�
=q����B����i��@���?�=qA�p�C !H                                    BxwW�  �          @�p��\(�@�\)��
=�0z�B�=q�\(�@��R?��\A>�RB��                                     BxwfH  �          @�ff�_\)@�Q쿂�\��
B����_\)@��?�
=AV{B�                                    Bxwt�  �          @ƸR�mp�@��R�u�p�B����mp�@�  @ffA�z�C�=                                    Bxw��  �          @ƸR�r�\@�p������\)B�Q��r�\@�p�@
=qA��C��                                    Bxw�:  �          @���X��@���>L��?���B�
=�X��@��H@!G�Aď\B��H                                    Bxw��  �          @�(��s33@���>��?�\)C c��s33@�@Q�A��\CW
                                    Bxw��  �          @���p  @��>���@5C ff�p  @�G�@�RA��C޸                                    Bxw�,  �          @�G��^{@�=q?��A33B��=�^{@qG�@K�A���C��                                    Bxw��  �          @\�^�R@�z�?fffA	�B��f�^�R@x��@EA�33C��                                    Bxw�x  �          @����1G�@�\)>��R@9��B�q�1G�@�{@6ffA���B�=q                                    Bxw�  �          @�33�E�@�Q�>aG�@�B�R�E�@�G�@*�HA�G�B�Q�                                    Bxw��  �          @����b�\@�Q�>�{@L(�B�z��b�\@�Q�@*=qA�\)C��                                    Bxwj  �          @�p��hQ�@�ff?   @�ffB�u��hQ�@�(�@1�A֏\CO\                                    Bxw  �          @Å�mp�@�=q?��@�ffB�33�mp�@\)@0��A�
=C�                                    Bxw$�  �          @���e@�33>��@xQ�B���e@��\@)��A��CaH                                    Bxw3\  �          @Å�e�@�ff>��?�
=B��q�e�@�G�@{A��C�
                                    BxwB  �          @�  ��{@��?(��@�=qC����{@Z�H@%A�z�C�                                    BxwP�  �          @�p����@i��?��AN{C^����@&ff@8Q�A�33C�                                    Bxw_N  �          @�  ����?��
@  A���C����?   @4z�B{C-�\                                    Bxwm�  �          @�33�\)��Q�@xQ�B0ffC9.�\)�
=q@O\)BG�CPs3                                    Bxw|�  �          @�Q���\)��p�@tz�B'��C8�3��\)���@L(�B�RCN��                                    Bxw�@  �          @��R�����p�@x��B+�HC9����
�H@P  B
CO��                                    Bxw��  T          @�p���=q���H@z=qB.z�C:���=q�33@L��B	��CQh�                                    Bxw��  �          @�����G���@Z=qB  C:����G���@0  A�CM�)                                    Bxw�2  �          @���~{=#�
@b�\B&��C3\)�~{��33@HQ�B=qCJ��                                    Bxw��  �          @�
=�P  ?�33@u�B=�C�R�P  ���@���BJ�RC>xR                                    Bxw�~  �          @�p���=q�n{@ffA�C>n��=q��Q�?�{A��\CH�3                                    Bxw�$  �          @����
�fff@��A�(�C=�����
��\)?ǮA~�HCH{                                    Bxw��  �          @�{��G��5?��
A.�\C;����G���\)?\)@�ffC?��                                    Bxw p  �          @�
=��z�Y��?��RA�C=�\��zῺ�H?fffA z�CD�
                                    Bxw  �          @��
��
=���?�p�A�Q�CC\��
=���?(��@�G�CH�                                     Bxw�  �          @�33���
���?�ffA���CJG����
�!�?+�@�G�CP��                                    Bxw,b  �          @����\)��(�?��Aep�CJ����\)�
=>L��@p�CNu�                                    Bxw;  �          @������
�ٙ�?fffACGB����
������Q�CI�=                                    BxwI�  �          @�����G��\)?�33Al(�C:��G���z�?uA ��C@h�                                    BxwXT  T          @��R��33��z�?\(�Ap�C7���33�+�?(�@�\)C;{                                    Bxwf�  �          @�������?8Q�@�33CE�\�������#�
�޸RCG=q                                    Bxwu�  �          @�p���{��p�>���@`��CC�3��{���H��(�����CC�3                                    Bxw�F  �          @�{��z��p��#�
�#�
CF�
��z��G��Y���  CDW
                                    Bxw��  �          @�ff���	���
=q��{CK������\)��G�����CF0�                                    Bxw��  �          @�z�����
�H��(���(�CL
=�����Q쿷
=�x  CG&f                                    Bxw�8  �          @�  ��{��(������R=qCK�)��{����Q���p�CB�                                     Bxw��  �          @�(���{�Y���0  �  C?k���{?
=�4z���\C+�                                    Bxw̈́  �          @�\)�]p�����dz��7�\C6}q�]p�?����L(���C&f                                    Bxw�*  �          @��R�&ff>\)��z��g\)C0�)�&ff@Q��c33�:(�C�
                                    Bxw��  �          @���Z=q=#�
�_\)�6�C3@ �Z=q?޸R�A��z�C�                                    Bxw�v  �          @�������?�33��=q��Q�CL�����@p��z��ᙚC�                                    Bxw  �          @�{��p�?#�
�33�ŅC+����p�?Ǯ��p���(�C ��                                    Bxw�  �          @�\)���>\)������z�C2=q���?�G��\��
=C'��                                    Bxw%h  �          @���G������\���C9s3��G�?B�\�����=qC)��                                    Bxw4  �          @�����z��G����陚C:��z�?B�\�  ���\C)��                                    BxwB�  �          @�����녿��ÿz���\CA.��녿&ff���
�I��C<{                                    BxwQZ  �          @����%�>�(��G����C*���%�?�{��ff���C+�                                    Bxw`   �          @�(�?��@�p���(��_�B�p�?��@�p�?�(�A^�\B�u�                                    Bxwn�  �          @��
?(��@�
=��p���=qB�?(��@��H?��\A<z�B�k�                                    Bxw}L  �          @���u@��H��ff�8  B�k��u@�ff?�=qA��RB�                                      Bxw��  �          @���Q�@���'���z�B�
=�Q�@�(�������B�#�                                    Bxw��  �          @��R�4z�@�z�u�$z�B�3�4z�@��@�
A��\B�\)                                    Bxw�>  �          @�ff�0��@�  ���H�W�
B�Ǯ�0��@�ff?�33Az�\B��                                    Bxw��  �          @��?aG�@z=q�p  �,p�B��q?aG�@������b�RB�
=                                    BxwƊ  �          @�\)@z�@=p��c33�1��B\(�@z�@�=q�У���z�B��
                                    Bxw�0  �          @��\@%�@u�#�
��B`=q@%�@�33��{�k�Br��                                    Bxw��  �          @�z�@G�@{��6ff� ffBo��@G�@�=q�
=q����B��                                    Bxw�|  �          @�33@p�@�
=�.{����Bn�
@p�@�Q쾔z��<(�BG�                                    Bxw "  �          @�{@��@�p��*�H��(�Bv��@��@�p��\)��Q�B�p�                                    Bxw �  �          @�\)@=q@��,����33Bv
=@=q@�{�#�
����B�=q                                    Bxw n  �          @��
@#33@�=q�333��Bg��@#33@���
=��=qBzz�                                    Bxw -  �          @���@��@��������Bxp�@��@��\>W
=@	��B��{                                    Bxw ;�  �          @��
@�R@�z������Br
=@�R@�
=>\)?���B}�H                                    Bxw J`  �          @��@   @�{������Brp�@   @��>aG�@G�B}p�                                    Bxw Y  �          @�(�?ٙ�@���\)��Q�B�z�?ٙ�@�33>B�\?�B��3                                    Bxw g�  �          @�33?�G�@�(��޸R���B�p�?�G�@�=q?��
A+33B�aH                                    Bxw vR  �          @���?�p�@��׿�������B��?�p�@�\)?��A%B���                                    Bxw ��  �          @�=q@%@��Ϳ�\���Bx�@%@�z�?fffAffB|��                                    Bxw ��  �          @���@0��@�����H�jffBqp�@0��@�{?�33A733Bs�                                    Bxw �D  �          @���@Q�@�ff��(���{B�k�@Q�@��?��A+�B���                                    Bxw ��  �          @�ff?���@�=q�z�����B�p�?���@���?Y��A  B�B�                                    Bxw ��  �          @�p��u@�G��!��̸RB�W
�u@��H>�G�@���B�8R                                    Bxw �6  �          @�=q?���@�p�����(�B�?���@���?�G�AH��B�u�                                    Bxw ��  �          @�녿aG�@��R�&ff����BŔ{�aG�@���@�A�33B�                                    Bxw �  �          @��\�h��@�{�Y����\B�.�h��@��@ffA�33B�L�                                    Bxw �(  �          @��\��=q@��׿z�����B�L;�=q@�G�@�A�33B�Ǯ                                    Bxw!�  �          @�\)<#�
@��ͿTz��B��)<#�
@���@ffA���B��)                                    Bxw!t  �          @�녿�@��H?^�RA
�HBˏ\��@���@c33B�BѨ�                                    Bxw!&  �          @�Q��\)@�p�?���A+�B��
�\)@|(�@`��B��B�.                                    Bxw!4�  �          @�  �Ǯ@�?��@�  B��=�Ǯ@��@\(�B�HB�33                                    Bxw!Cf  �          @��Ϳ��@��>B�\?�B��׿��@�\)@FffA���B��                                    Bxw!R  �          @��\=�\)@��þ�=q�-p�B��{=�\)@�z�@*=qA���B�k�                                    Bxw!`�  �          @�33?�@�(��}p���
B���?�@��
?�(�A��B��                                     Bxw!oX  �          @�\)?#�
@��
�Q����
B�� ?#�
@��@\)A��\B���                                    Bxw!}�  �          @��\?O\)@�z῕�9��B���?O\)@��R?���A��B�#�                                    Bxw!��  �          @�
=?G�@�G���  �!�B���?G�@�G�?�A�  B�8R                                    Bxw!�J  �          @�(�?z�H@��R�Y�����B�� ?z�H@��@��A��B�B�                                    Bxw!��  �          @�33�333@�{?
=@�z�B�{�333@��
@W�BG�B�L�                                    Bxw!��  �          @�=q�\)@���=��
?Tz�B��׿\)@�\)@6ffA�
=B��{                                    Bxw!�<  �          @��H�k�@��ü���33B��)�k�@���@0��A��HB���                                    Bxw!��  �          @��
�h��@��R?G�A�B��h��@��@S33B\)B��                                    Bxw!�  �          @�33��(�@�G�?+�@�p�B�33��(�@�\)@O\)BQ�B�z�                                    Bxw!�.  T          @��
��Q�@��\?#�
@�p�BӮ��Q�@��@G
=Bp�B�\                                    Bxw"�  �          @����33@�\)?�  A-p�Bب���33@r�\@XQ�B
=B�                                    Bxw"z  
�          @�ff����@���@�
A��
B�W
����@E�@�{BLffB�aH                                    Bxw"   "          @��R�@  @���@��A�{B�ff�@  @E@�BY\)B�Q�                                    Bxw"-�  �          @��R�
=@�
=?�Ag\)B�8R�
=@�Q�@\)B2p�Bģ�                                    Bxw"<l  �          @�\)��\@�
=@�\A���B�B���\@U@��BW  Bŀ                                     Bxw"K  �          @��
�^�R@��\@-p�A���B�k��^�R@333@���Bj  B֔{                                    Bxw"Y�  �          @�
=����@�(�@0  A�\B�  ����@3�
@��HBg��Bߨ�                                    Bxw"h^  �          @��H��  @��?��HA��HB�.��  @Z=q@���BI  BԳ3                                    Bxw"w  �          @�Q���@�ff@I��B�B�\���@p�@�33B�HB�#�                                    Bxw"��  �          @��;��
@�=q@,(�A���B�33���
@@��@�z�Bh�HB��                                    Bxw"�P  �          @��H=�@�
=?��RA��B�aH=�@_\)@��BMp�B�{                                    Bxw"��  �          @�ff��
=@��@   A�Q�B���
=@c�
@�p�BK��B�k�                                    Bxw"��  �          @��R��
=@�  @��A�33B�{��
=@Z=q@��BT�B���                                    Bxw"�B  �          @������R@���?uA$Q�B��׾��R@�=q@aG�B"��B���                                    Bxw"��  �          @����\)@�\)>8Q�?�{B��쾏\)@�(�@<(�Bz�B��H                                    Bxw"ݎ  �          @��>8Q�@�z�?h��A�HB�aH>8Q�@�z�@i��B��B�W
                                    Bxw"�4  �          @�?\)@�\)?O\)Ap�B�W
?\)@��@_\)B��B�33                                    Bxw"��  �          @��R�
=@���@Y��B{B�Ǯ�
=@(�@��B�B�B�8R                                    Bxw#	�  
�          @�����@��@-p�A��B����@0��@���Bo��B�33                                    Bxw#&  �          @�z�>8Q�@�
=@C�
B�
B��=>8Q�@   @�G�B~33B��{                                    Bxw#&�  "          @��>��
@�G�@UB{B�aH>��
@�H@��B��{B���                                    Bxw#5r  
�          @�G�>���@�@7�A�G�B�Q�>���@#�
@��Bw�B�p�                                    Bxw#D  �          @�
=��@���?�G�AU�B�G���@y��@qG�B/{BĔ{                                    Bxw#R�  
�          @�  ��Q�@��?�{A�ffB��;�Q�@k�@��B>��B�{                                    Bxw#ad  "          @�녽u@��R?�\)A��B�G��u@a�@���BI��B��f                                    Bxw#p
  "          @�=q>��R@�{@\)A��
B�Ǯ>��R@>{@��BeffB�
=                                    Bxw#~�  
�          @��R?�z�@��@%�A�=qB�
=?�z�@+�@�33BfG�B�B�                                    Bxw#�V  "          @�{?G�@�p�@AG�BG�B�#�?G�@{@��BzQ�B��                                    Bxw#��  "          @��H?�@��@]p�B��B���?�@\)@��B��\B�8R                                    Bxw#��  T          @���?(�@��@n�RB$33B���?(�?�p�@�(�B�=qB�Q�                                    Bxw#�H  �          @�Q�J=q@��?\(�A=qB�녿J=q@��
@]p�B{BɸR                                    Bxw#��  �          @�=q����@�{?��
A+�B�uþ���@�z�@j�HB%�RB��                                    Bxw#֔  �          @�  �E�@��\>�z�@C�
B�p��E�@�@>�RB
=BǸR                                    Bxw#�:  "          @��ý���@���?���A5��B�
=����@���@w�B(�B��3                                    Bxw#��  �          @�����
@�p�?W
=A\)B�zἣ�
@�{@hQ�B=qB���                                    Bxw$�  T          @����.{@���?���A@  B�uþ.{@��@{�B+\)B���                                    Bxw$,  �          @�Q�   @���?�A��B����   @l��@�33BEz�BÊ=                                    Bxw$�  �          @����5@���=#�
>�Q�B�G��5@��\@;�A�BĮ                                    Bxw$.x  �          @��
�=p�@��׿�����B=�=p�@�Q�@��A�(�B��                                    Bxw$=  �          @�z�   @��R��  ��HB�\�   @��@z�A���B���                                    Bxw$K�  
�          @�=q�L��@������RB��L��@�z�@8��A�(�BƔ{                                    Bxw$Zj  T          @���+�@�z������B��=�+�@�z�@��A�B���                                    Bxw$i  T          @��H�u@�
=>\)?�Q�B��)�u@��H@Dz�BffB��q                                    Bxw$w�  "          @�
=>�
=@�ff>.{?޸RB�� >�
=@���@E�BffB��                                    Bxw$�\  �          @�Q�?�@��H�����9��B�\?�@�ff@'�A�(�B��=                                    Bxw$�  
�          @�?�@��@��A�z�B��?�@X��@�p�BQ�B��H                                    Bxw$��  
(          @���>�p�@��
@%A�(�B���>�p�@C�
@�33Bfz�B�L�                                    Bxw$�N            @�>���@�z�@=p�A�p�B�=q>���@8Q�@�{BsQ�B���                                    Bxw$��  �          @��R�#�
@�@(�A��B�Lͼ#�
@X��@�z�B]  B��                                     Bxw$Ϛ  
�          @�p��\)@��@(�A��
B��f�\)@e�@�ffBR�B�u�                                    Bxw$�@  �          @��   @���@(��A�B��)�   @J=q@�\)Be�B�
=                                    Bxw$��  T          @��ÿ@  @���@#33A�(�B��H�@  @Tz�@�
=B^ffB͔{                                    Bxw$��  �          @�
=�s33@���?�Q�A�{B�W
�s33@p  @���BEB�G�                                    Bxw%
2  �          @�p��z�@�\)?��
AG�B��׿z�@�  @���B-Q�Bã�                                    Bxw%�  �          @��\�L��@��?p��A��B�
=�L��@�@qG�B!\)B�=q                                    Bxw%'~  "          @���  @�z�.{��\B�{��  @�p�@0��A�(�B���                                    Bxw%6$  
�          @����L��@�  �z��Ǚ�B�\�L��@�\)?
=@�(�B���                                    Bxw%D�  �          @�����@��'
=���B�z���@�=q>��R@L(�B�{                                    Bxw%Sp  �          @��;���@�{�!G���(�B�z����@��@G�A��B�B�                                    Bxw%b  
�          @��þ��H@��Ϳ�(�����B�Q���H@�z�?���A0��B��H                                    Bxw%p�  
Z          @�(����
@�����
=��33B�z���
@�
=@ ��A˙�B�W
                                    Bxw%b  
�          @�33�%@��������B�aH�%@�
=@	��A�ffB�                                    Bxw%�  "          @���(�@�=q�z���G�B�Ǯ�(�@��@  A�G�B䙚                                    Bxw%��  
�          @����'�@�p��\)���B��'�@�  @#33A�
=B�\)                                    Bxw%�T  �          @���5�@���\)�0  B�B��5�@��\@(�A��HB�aH                                    Bxw%��  "          @���#�
@����.{��\)B�\�#�
@��
@&ffAљ�B���                                    Bxw%Ƞ  "          @��ÿ�=q@��
����
=Bγ3��=q@��
@333A�z�Bң�                                    Bxw%�F  "          @����@��������(�B�G���@��@!�A�G�B�\)                                    Bxw%��  �          @���W
=@��þ.{��z�BĀ �W
=@�G�@6ffA�\)B��                                    Bxw%��  
�          @����@��Ϳs33�(�B�(���@��\@
=A��
B���                                    Bxw&8  
�          @����@�  ����*�HB���@���?�z�A�G�B�k�                                    Bxw&�  "          @\����@�<��
>L��B�(�����@�33@=p�A�z�B��=                                    Bxw& �  "          @��
@ff@��@   A�G�B��@ff@G�@��BO�B`�                                    Bxw&/*  T          @�33?��H@�@��A�
=B�
=?��H@`  @�{BI(�B�                                      Bxw&=�  "          @��?��R@��
?�{A�33B�.?��R@u@�  B@p�B�W
                                    Bxw&Lv  T          @�  ?O\)@�z�?�z�A�=qB��?O\)@u�@��BE=qB�.                                    Bxw&[  �          @�\)?��@�p�@��A�
=B�\)?��@`  @�{BO��B�\                                    Bxw&i�  
�          @�Q�?�  @��R@K�A�33B���?�  @%@�G�Bu��B���                                    Bxw&xh  "          @���#�
@�{@mp�Bz�B�33�#�
?�\)@�
=B��B��                                    Bxw&�  
�          @�=q�8Q�@�  @l��B\)B�\)�8Q�@��@��
B�.B�aH                                    Bxw&��  T          @������@�\)@0  A�=qB�𤿇�@C33@�=qBf�B�k�                                    Bxw&�Z  �          @�G����@���@&ffA�BЀ ���@J�H@��RB]  B��f                                    Bxw&�   "          @�녿��@�p�@ffA���B�B����@Z=q@�=qBSQ�B��                                    Bxw&��  
�          @�33�s33@���?�=qA�  B�
=�s33@xQ�@�  BA�B�u�                                    Bxw&�L  �          @�G���
=@��
?@  @��
B��f��
=@��H@tz�B��Bπ                                     Bxw&��  "          @�
=��p�@�  ?\)@���BϮ��p�@��H@eBffB���                                    Bxw&�  T          @��Ϳ�ff@�
=?�@�  B̞���ff@��\@c�
B�RB�8R                                    Bxw&�>  �          @��Ϳ�(�@��R=#�
>�33Bϣ׿�(�@�=q@HQ�A���B�=q                                    Bxw'
�  
�          @�\)���@�  =�G�?z�HB̮���@�=q@L��A���B�                                      Bxw'�  �          @�{�ٙ�@�33?�G�A�\B�uÿٙ�@�
=@y��B�Bݮ                                    Bxw'(0  "          @�{��\)@��R>�?��RB�uÿ�\)@���@Mp�A�Q�B���                                    Bxw'6�  
�          @�(��ff@����z��.{B�L��ff@���@.�RA�{B�8R                                    Bxw'E|  
�          @����ff@���<�>��RB��ff@��R@9��A��HB�                                      Bxw'T"  T          @�
=��\@�\)>\)?���B�Q���\@��@=p�AB�Ǯ                                    Bxw'b�  T          @��G�@���<��
>aG�B��G�@���@(Q�A��B�(�                                    Bxw'qn  �          @�  �.{@�=q>���@uB�#��.{@��H@G
=A�  B�.                                    Bxw'�  
�          @�\)�z�@���?8Q�@��B�\�z�@��@_\)Bp�B��H                                    Bxw'��  �          @��׿���@�ff>�@��B�����@��
@W�B�\B߸R                                    Bxw'�`  T          @�  �
�H@���?&ff@�\)B��)�
�H@�(�@Z�HB  B��                                    Bxw'�  "          @�G����@�\)?J=q@�B�W
���@���@aG�B�B��                                    Bxw'��  �          @���{@�{?�z�A\(�Bٮ��{@z�H@���B+��B�3                                    Bxw'�R  T          @����Q�@��\�k����Bݽq�Q�@�z�@,��A��B�\                                    Bxw'��  �          @��{@�  =#�
>\B����{@�@8��A�p�B���                                    Bxw'�  
!          @�{��@��>�@�z�B�k���@�{@P��B�\B�ff                                    Bxw'�D  	g          @�=q�3�
@�ff?��AS�B��)�3�
@n{@x��B=qB�8R                                    Bxw(�  �          @�=q�"�\@��\?�\)AP��B���"�\@u@|(�B"(�B��f                                    Bxw(�  �          @\�$z�@��?��A{�B���$z�@hQ�@�(�B+��B��=                                    Bxw(!6  �          @�\)�{@�33?���A`z�B�{�{@tz�@���B)�B�W
                                    Bxw(/�  �          @��
��33@�
=?��
AJ�\BՅ��33@�Q�@|(�B)�B�3                                    Bxw(>�  �          @�Q쿎{@�\)?���A]B��f��{@~{@���B1p�B�B�                                    Bxw(M(  �          @��ÿTz�@��H?�
=A=�B��Tz�@��@z=qB*Bʞ�                                    Bxw([�  �          @�G���@�
=?!G�@��B��)��@���@h��B�B��=                                    Bxw(jt  �          @�=q��p�@\?�A+\)B��)��p�@��\@��B$��B�                                      Bxw(y  T          @�=q��G�@�ff?�
=A�{B�녿�G�@�  @���B9Q�B���                                    Bxw(��  �          @��Ϳ�33@�33@G�A��RB�33��33@tz�@�\)BM33B�p�                                    Bxw(�f  �          @ȣ׿333@��
@z�A�=qB��=�333@|(�@�=qBI�B�\                                    Bxw(�  �          @ə���z�@���?��
A��B�zᾔz�@�
=@���B>�HB���                                    Bxw(��  �          @Å=��
@��R?��AE�B�u�=��
@�p�@�
=B.��B��                                    Bxw(�X  �          @�(�?k�@�  ?O\)@�B��?k�@��R@s�
B�HB���                                    Bxw(��  �          @�z�?��H@�����
�O\)B�?��H@��H@5A��B���                                    Bxw(ߤ  �          @�G�@33@��;�����HB�ff@33@�{@+�A���B�Ǯ                                    Bxw(�J  �          @�\)@ ��@��\�5��33B�u�@ ��@�z�@�\A�Q�B�W
                                    Bxw(��  �          @��\?��H@��L����(�B�B�?��H@��@��A�{B��{                                    Bxw)�  �          @�z�?�G�@�  �z�H�2=qB�Ǯ?�G�@�=q?У�A���B�z�                                    Bxw)<  �          @�\)�8Q�@p�@l(�B5�
C���8Q��@���B`�C4�{                                    Bxw)(�  T          @�����@   @�RBffC8R��?Y��@Z�HBa{C{                                    Bxw)7�  
�          @�녾8Q�@��H?��
AuG�B��R�8Q�@~�R@��RB:\)B�8R                                    Bxw)F.  �          @�p��L��@�(�?��HA�G�B�  �L��@z�H@�z�B@��B��q                                    Bxw)T�  �          @�ff�B�\@�ff@(�A���B��B�\@X��@�p�B]p�B�W
                                    Bxw)cz  �          @���>u@�33?���A]�B�  >u@�=q@��\B433B�#�                                    Bxw)r   �          @���=�\)@�z�?���A@  B��==�\)@�ff@|��B-  B�                                    Bxw)��  �          @����u@��
?�\)A[�B�Lͽu@��H@��HB3��B���                                    Bxw)�l  �          @�33>�ff@�
=?�{A.�RB���>�ff@�=q@z�HB(33B�
=                                    Bxw)�  �          @���?O\)@�\)?��A&�\B��f?O\)@�33@xQ�B%{B��f                                    Bxw)��  �          @��R?&ff@�\)?�Q�A`(�B���?&ff@���@�ffB3��B�{                                    Bxw)�^  �          @�\)?\@�p�?^�RAffB�
=?\@���@k�Bp�B��                                    Bxw)�  �          @��@N{@�Q�\�k�Be(�@N{@�\)@G�A���BYQ�                                    Bxw)ت  �          @��H@;�@�G���p��j=qBo\)@;�@�  @�\A�
=Bc�
                                    Bxw)�P  T          @��@!�@�
=>\)?��B�W
@!�@�z�@4z�A���Bp\)                                    Bxw)��  �          @���?���@��?�(�A�
=B�� ?���@U@��\BJ��B�p�                                    Bxw*�  �          @�33��(�@���@l��B%ffBճ3��(�?\@��B�8RC ��                                    Bxw*B  �          @��׿��
@�@�p�BmffC�ÿ��
�(�@�33B��CF��                                    Bxw*!�  �          @�ff��@fff@x��B4\)B�  ��?���@�{B�33C�                                    Bxw*0�  �          @��\�У�@|(�@|��B+=qB�
=�У�?���@�p�B��{C��                                    Bxw*?4  �          @��ÿ���@fff@�33B5�RB�R����?u@��
B�CB�                                    Bxw*M�  �          @��\��Q�@��H@s33B"�\B��ÿ�Q�?��
@��B��C	ٚ                                    Bxw*\�  �          @�G����@h��@��\B4�\B�z���?�  @��
B��C�=                                    Bxw*k&  �          @��H����@���@UB��B�aH����@��@�ffB{�RB���                                    Bxw*y�  �          @��R�(�@��\?xQ�AQ�B��{�(�@���@mp�B"B�\)                                    Bxw*�r  �          @�p��@  @��
?�Q�Am�B���@  @vff@�Q�B6�B��                                    Bxw*�  �          @�z�z�H@�=q@,(�A噚B��)�z�H@/\)@���BkQ�B�=q                                    Bxw*��  T          @�ff���@��R?�p�A�G�B̸R���@[�@�z�BJ{B���                                    Bxw*�d  �          @�33���R@�=q@ffA�Q�B�8R���R@^{@���BL
=B�W
                                    Bxw*�
  �          @������@�
=@#�
A�\)BѨ�����@<(�@�Q�B_�RB�L�                                    Bxw*Ѱ  �          @�녿�(�@��\@=qA�ffB�LͿ�(�@Fff@�{BWQ�B�                                    Bxw*�V  �          @�������@���@  A��\B�aH����@O\)@��\BP�B�
=                                    Bxw*��  �          @�\)���
@�33@A��B�uÿ��
@R�\@�p�BJp�B���                                    Bxw*��  �          @�\)��p�@�=q@G�A��Bٳ3��p�@S33@�33BEB�ff                                    Bxw+H  
�          @�\)��ff@���@{A�z�B����ff@K�@�Q�BO��B���                                    Bxw+�  T          @��׿\@���?��A�{B��\@l��@�z�B7�B�q                                    Bxw+)�  T          @�33�^�R@%@z�HB(p�CJ=�^�R>L��@�ffBUC0�q                                    Bxw+8:  �          @���c�
@N�R@u�B{C	Ǯ�c�
?J=q@�{BV\)C'n                                    Bxw+F�  �          @�\)�c�
@4z�@z�HB#{C�\�c�
>���@��BUffC-�{                                    Bxw+U�  T          @�p��'
=@L��@�
=B6��C5��'
=?\)@���B|�C'��                                    Bxw+d,  T          @�33�ٙ�@u@hQ�B#Q�B�׿ٙ�?�@��HB���C                                    Bxw+r�  "          @��׿���@�=q@N�RB{B�𤿙��@�\@�  B�W
B�                                    Bxw+�x  �          @�\)����@�(�@@  B�
B�LͿ���@p�@��\Bw(�B��                                    Bxw+�  �          @�ff�8Q�@��\@333A��B���8Q�@\)@�Q�Bu�B�=q                                    Bxw+��  �          @��u@�z�@&ffA�{B�aH�u@(��@��
Bj��B���                                    Bxw+�j  �          @�
=��=q@���@1G�A�  B���=q@(�@�ffBm�B�                                      Bxw+�  "          @��׿��@�Q�@FffB
z�B�
=���@�
@�33B{ffB���                                    Bxw+ʶ  T          @�  ����@�{?Y��AffB�B�����@��@r�\B33B�                                      Bxw+�\  
�          @����ff@�33?���A-B��
��ff@�{@~{B&�RB���                                    Bxw+�  T          @��þ�{@�Q�?�G�A�p�B��쾮{@�G�@�
=B?�B��R                                    Bxw+��  �          @�  >�33@�
=@0��A�(�B�  >�33@5@�Bo=qB�\                                    Bxw,N  �          @�\)?�=q@���@:=qA�33B���?�=q@'�@��RBq�B�(�                                    Bxw,�  �          @�z�?Y��@�{@E�A�ffB�z�?Y��@*=q@�Bv�B��=                                    Bxw,"�  �          @��?}p�@���@4z�A��B��?}p�@7�@�  Bk
=B��
                                    