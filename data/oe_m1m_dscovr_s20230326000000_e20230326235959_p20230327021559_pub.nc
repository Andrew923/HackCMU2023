CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230326000000_e20230326235959_p20230327021559_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-27T02:15:59.268Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-26T00:00:00.000Z   time_coverage_end         2023-03-26T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxq�o�  �          @�ff@��
����@ffA��\C��@��
�����Ǯ�X��C�y�                                    Bxq�~&  
�          @�@�(��n�R@+�A���C�L�@�(����H>�=q@�C�c�                                    Bxq���  �          @��H@��R�i��@\)A�C��{@��R���ýu��\C���                                    Bxq��r  �          @Ϯ@�Q��Z�H?�{AC33C�w
@�Q��g��.{��33C���                                    Bxq��  T          @�{@��\)���\�8��C�,�@���
=�����=qC��{                                    Bxq���  T          @�\)@�33�"�\�k��
=C�� @�33��z������C���                                    Bxq��d  �          @ҏ\@���Y��?�Q�AS33C�.@���h�ÿ����C�AH                                    Bxq��
  
�          @�(�@�(��|(�@'�A�ffC�˅@�(���\)=�G�?xQ�C�1�                                    Bxq��  T          @ҏ\@�p��~{@ffA�  C���@�p����
����\)C���                                    Bxq��V  
�          @��@�
=�s33?�p�A�Q�C�L�@�
=���׾�33�E�C���                                    Bxq��  
�          @Ӆ@��{�?�(�A���C��q@����
��(��mp�C�:�                                    Bxq��  !          @�33@����u�?���A���C�Q�@�����\)������C��\                                    Bxq�H  
�          @�33@���s33@*=qA���C�e@����z�>aG�?��HC���                                    Bxq�-�  T          @�=q@��\�S�
?�33A���C�{@��\�s�
�����=qC�4{                                    Bxq�<�  T          @��@��
�dz�@ ��A���C��\@��
���H�L�Ϳ�p�C��\                                    Bxq�K:  "          @��@�(��-p�?��RA��C�'�@�(��Vff>u@
=C���                                    Bxq�Y�  
�          @У�@�G��8Q�?�=qA��\C�G�@�G��Z=q<��
>.{C�33                                    Bxq�h�  T          @�33@�G�����@��B  C�j=@�G��q�@"�\A��\C���                                    Bxq�w,  �          @�(�@�(��z�@~�RBC���@�(��w�@�A���C��=                                    Bxq���  �          @�(�@�p����@��
B�RC�)@�p���33@�\A�C�^�                                    Bxq��x  
�          @�z�@��H�+�@|(�B=qC��H@��H���?�A��HC�<)                                    Bxq��  
�          @�=q@���5@9��A�G�C�xR@���|(�?s33AffC�'�                                    Bxq���  "          @�=q@����_\)@��B��C��\@�����z�?У�A^=qC�n                                    Bxq��j  
Z          @׮@����C�
@�G�B(�C��{@������?�Ayp�C�n                                    Bxq��  T          @��@�  �*=q@|��Bp�C���@�  ��33?���A�(�C���                                    Bxq�ݶ  �          @�G�@�(��(Q�@q�B�\C�4{@�(���\)?�A�
=C�Ǯ                                    Bxq��\  	�          @ۅ@�  �@��@��RB"Q�C�` @�  ���@(�A��C�q�                                    Bxq��  �          @�G�@����L��@�p�B"��C���@�������@�
A��C�J=                                    Bxq�	�  T          @�=q@{��P��@���B&  C�0�@{�����@
=A�=qC���                                    Bxq�N  �          @�=q@dz��P��@���B2�C�Ф@dz���G�@�A��\C�H                                    Bxq�&�  T          @�(�@b�\�Z=q@��B1�C��@b�\��{@�A��HC���                                    Bxq�5�  �          @��@@������@�ffB,ffC�k�@@������?�\)A{�
C��f                                    Bxq�D@  �          @��
@Vff�\)@��RB"\)C���@Vff��  ?�
=Ab�HC��                                    Bxq�R�  T          @��@   ��z�@�B �C�,�@   ����?��A/�C�޸                                    Bxq�a�  �          @�z�@0����Q�@���B)�C�z�@0����=q?ٙ�Ad��C�:�                                    Bxq�p2  �          @�p�@N{��  @�(�B((�C�k�@N{���H?���At��C�o\                                    Bxq�~�  
�          @��
@Tz��qG�@��
B*p�C���@Tz���z�?�Q�A�  C�@                                     Bxq��~  T          @�(�@����G
=@��
Bp�C��f@������?�{A{�C���                                    Bxq��$  �          @�z�@�\)�K�@s33B��C�\@�\)��ff?ǮAR{C��
                                    Bxq���  
�          @��@��R�B�\@mp�B33C�@ @��R��G�?ǮAPz�C��                                    Bxq��p  �          @�@���9��@|��Bz�C��
@����G�?�=qAuC�y�                                    Bxq��  "          @��
@�  �I��@s�
B�C�7
@�  ��{?�=qAU�C��{                                    Bxq�ּ  
(          @�33@���L(�@~{B��C���@����G�?ٙ�Af=qC�Ǯ                                    Bxq��b  "          @ۅ@����?\)@e�A�  C��H@�����p�?�p�AG33C�P�                                    Bxq��  
�          @��@���Mp�@�
=B!z�C�33@�����\@�A�
=C��                                    Bxq��  "          @޸R@����C�
@�\)B,Q�C�b�@������H@�HA��C�/\                                    Bxq�T  T          @�
=@��\�7�@�33B&�C�:�@��\���@=qA��\C���                                    Bxq��  �          @߮@����k�@`��A��C�(�@�����\)?��A�C��                                    Bxq�.�  �          @��@�=q�\)@I��A�C�+�@�=q����?
=@�=qC���                                    Bxq�=F  �          @�  @����p�@2�\A�ffC���@����Q�>L��?���C�3                                    Bxq�K�  �          @߮@����qG�@c33A�33C�z�@�����=q?�ffA
�\C�C�                                    Bxq�Z�  T          @�Q�@����j�H@fffA�=qC�� @�����Q�?�33Az�C�q�                                    Bxq�i8  
�          @��@�33����@=qA�C���@�33������R�!G�C�                                    Bxq�w�  T          @�{@�Q���\)@
=A��RC�\)@�Q����������0  C���                                    Bxq���  �          @�@�(���(�@��A�(�C�� @�(���(����H��33C�
=                                    Bxq��*  �          @���@�  ��  @z�A�{C��R@�  ��(��8Q�����C��                                     Bxq���  
          @�\)@�{���
?��
AK
=C�u�@�{��p���\)�6{C�W
                                    Bxq��v  T          @��
@����8Q�@a�A�{C�#�@�������?�G�AL��C��=                                    Bxq��  T          @ۅ@��H�A�@h��B
=C���@��H��\)?��
APQ�C��R                                    Bxq���  
�          @��@�\)�!�@�p�B�C��@�\)���\@\)A�(�C���                                    Bxq��h  	�          @�33@������@�33BffC���@�����
=@�RA�{C��)                                    Bxq��  �          @�(�@����2�\@�G�B�RC��3@������@   A��C�^�                                    Bxq���  
�          @�33@�Q��l��@`  A�  C�XR@�Q���\)?���A��C�R                                    Bxq�
Z  "          @�  @���j=q@S33A�ffC�Ǯ@�����H?k�@��\C�˅                                    Bxq�   T          @�G�@�  �l(�@`��A���C�Z�@�  ��
=?�=qA�RC�
                                    Bxq�'�  T          @��@���QG�@aG�A�33C��@�����
?��A1p�C��                                    Bxq�6L  T          @ٙ�@����r�\@W�A�C��@������?k�@��C�                                      Bxq�D�  
�          @ۅ@�G���33@[�A�Q�C�(�@�G�����?O\)@���C��=                                    Bxq�S�  �          @ڏ\@�G��z�H@c33A�33C���@�G���{?�  A\)C�Ǯ                                    Bxq�b>  �          @���@����k�@N�RA�z�C���@�����=q?\(�@��C��)                                    Bxq�p�  
�          @أ�@����|(�@G
=Aۙ�C�w
@�����\)?!G�@��C��                                    Bxq��  �          @�  @�  ��ff@C�
Aأ�C���@�  ��z�>��
@,��C�N                                    Bxq��0  �          @׮@C33���R@hQ�B�C�p�@C33���?+�@��C���                                    Bxq���  �          @�G�@b�\���@N�RA��HC�aH@b�\��
=>��R@*�HC��q                                    Bxq��|  �          @�=q@QG���ff@VffA�C���@QG���
=>��R@(��C�\)                                    Bxq��"  
�          @�33@c33���R@`��A�ffC�~�@c33��33?z�@�33C���                                    Bxq���  
�          @���@e���=q@n{BC�
=@e����\?Tz�@�ffC���                                    Bxq��n  �          @�ff@a���  @g�A�p�C�L�@a���{?&ff@�z�C�]q                                    Bxq��  
�          @�p�@c�
���@vffB\)C�0�@c�
���\?z�HA�RC��3                                    Bxq���  �          @�@q���ff@l��B�\C�'�@q���
=?aG�@�33C��
                                    Bxq�`  
          @�
=@k���p�@h��A���C�)@k���(�?8Q�@���C�f                                    Bxq�  
�          @�\)@qG���\)@^�RAC�G�@qG���33?\)@��HC�g�                                    Bxq� �  
�          @�
=@w����@Tz�A�C��H@w�����>��@Y��C��                                    Bxq�/R  	�          @�\)@�����  @VffA噚C�U�@������\?�@���C�O\                                    Bxq�=�  "          @���@�{���R@]p�A�C���@�{��33?.{@�G�C�e                                    Bxq�L�  T          @�  @�{��G�@QG�A��C�Z�@�{��=q>�@|(�C�xR                                    Bxq�[D  
�          @�Q�@�ff���@J�HAأ�C�Q�@�ff��G�>Ǯ@K�C���                                    Bxq�i�  �          @߮@���{@*�HA�C�AH@���<��
>#�
C��                                    Bxq�x�  "          @���@����tz�@!�A�  C�u�@�����=q>aG�?�C���                                    Bxq��6  
�          @�z�@�(��q�?�(�A�C�p�@�(���Q�u��(�C��\                                    Bxq���  "          @ڏ\@��H�j=q@�A���C�˅@��H��Q�    <��
C��R                                    Bxq���  
�          @�\)@�ff����@4z�A���C�o\@�ff����>�p�@C33C���                                    Bxq��(  �          @�=q@�Q����@P  A�G�C���@�Q�����?8Q�@��C�aH                                    Bxq���  �          @�G�@�����@s33B{C��@�����\?�{A��C�/\                                    Bxq��t  
�          @���@}p���\)@~{B	C��=@}p����?�G�A%C�xR                                    Bxq��  �          @�  @\)���R@z�HB=qC��)@\)���?�p�A"ffC���                                    Bxq���  �          @�\)@����{@n{B ��C�b�@����  ?���A�C���                                    Bxq��f  T          @���@z�H����@�ffB33C��@z�H���?���AO\)C�xR                                    Bxq�  �          @�  @�=q����@p  B�\C���@�=q���H?��A
�HC��                                    Bxq��  �          @ᙚ@tz����\@~�RB
ffC���@tz���  ?��RA"�\C��                                    Bxq�(X  T          @�33@�����@���BffC���@������?�
=A:=qC�ff                                    Bxq�6�  T          @��@����y��@��B\)C���@�����
=?�{APQ�C��                                    Bxq�E�  
�          @�@�Q�����@u�B�C��@�Q���?�  A ��C�ٚ                                    Bxq�TJ  "          @�33@�����R@fffA�ffC�*=@�����R?z�H@�p�C�|)                                    Bxq�b�  "          @�@�
=��  @J�HA�{C��@�
=����?
=@��C��                                    Bxq�q�  "          @��H@����@G�AиRC���@����?z�@�
=C�o\                                    Bxq��<  T          @�G�@��
��ff@7
=A���C��@��
����>k�?��C��3                                    Bxq���  
          @�G�@C�
���
@�(�B(Q�C�G�@C�
����?�A�33C��3                                    Bxq���  l          @�@R�\���@���B*=qC��H@R�\��@ffA��C��                                    Bxq��.  �          @�=q@6ff��=q@�Q�B5  C��H@6ff��G�@�A���C���                                    Bxq���  �          @߮@w
=�~�R@���BG�C��@w
=����?˅AS�
C�w
                                    Bxq��z  �          @��@��
�`  @AG�A��C��R@��
��G�?\(�@�z�C�8R                                    Bxq��   "          @�ff@����e@l(�B �C�%@������R?��A9�C��\                                    Bxq���  �          @�{@��R�i��@uB�RC�Z�@��R���\?��RAG
=C���                                    Bxq��l  
Z          @�{@����w
=@u�B�\C��@�����Q�?���A8  C��f                                    Bxq�  
�          @�(�@~�R�P  @���B%\)C�aH@~�R���\@�\A�  C��                                    Bxq��  �          @�@p  �Z�H@�{B*�C�˅@p  ���@ffA���C��)                                    Bxq�!^  
�          @�p�@Z�H�W�@�
=B8{C��{@Z�H���@'�A�ffC�'�                                    Bxq�0  �          @�=q@^�R�W
=@�z�B:�
C�@^�R��
=@1G�A���C�:�                                    Bxq�>�  �          @��@G
=�c33@�{B>�HC��@G
=���@.�RA�
=C�g�                                    Bxq�MP  
�          @��@Z�H�n�R@���B8�C�:�@Z�H���
@.{A�ffC��                                    Bxq�[�  �          @��@n{�R�\@���B;�C�B�@n{���@=p�A�=qC�                                    Bxq�j�  T          @���@QG��Y��@���BF(�C���@QG���ff@FffA�=qC��                                    Bxq�yB  �          @�  @tz��HQ�@���B;��C�\)@tz����H@AG�A�C��                                    Bxq���  �          @�(�@S33�`��@�=qBDC�� @S33���@EA���C���                                    Bxq���  
�          @�@G��tz�@��BA��C��H@G���=q@;�A���C���                                    Bxq��4  �          @�Q�@Fff�{�@��\B@G�C�&f@Fff��@8��A���C�U�                                    Bxq���  T          @�G�@\(��\(�@�\)BFC�� @\(����\@QG�AΏ\C�E                                    Bxq�  �          @��@qG��S�
@�p�BA�HC�T{@qG���{@Q�AͅC���                                    Bxq��&  	�          @��H@i���P��@���BG
=C�{@i����ff@Z=qA�{C�O\                                    Bxq���  "          @��R@�ff�s33@�{B)��C��@�ff���
@(Q�A�=qC��=                                    Bxq��r  �          @���@�Q��J�H@��B:\)C��3@�Q�����@Tz�A�p�C��3                                    Bxq��  T          @��R@�����@��BDQ�C�s3@����
=@�  A��RC���                                    Bxq��  
�          @�G�@���  @�=qBA�C��@�����\@}p�A�Q�C��)                                    Bxq�d  
�          @��
@��1G�@�{B=�\C���@����R@e�A�
=C�/\                                    Bxq�)
  
�          AG�@{���  @k�A�\)C���@{���G�>�  ?�ffC���                                    Bxq�7�  "          Ap�@i����@hQ�A��HC�J=@i����p�>�?h��C��                                     Bxq�FV  "          @�\)@s�
��Q�@�=qB=qC�W
@s�
��p�?��@��
C�l�                                    Bxq�T�  
�          @��@s33���R@�\)B  C�h�@s33�ڏ\?�G�@�z�C���                                    Bxq�c�  	�          @�\)@�Q�����@qG�A�C�b�@�Q���(�>�@R�\C�)                                    Bxq�rH  �          @��R@�p����\@]p�A�{C�˅@�p�����>#�
?�z�C��\                                    Bxq���  "          @�@������@^�RAЏ\C���@����\)>W
=?�ffC��q                                    Bxq���  �          @��R@~{��=q@p��A��\C�*=@~{���>�(�@G�C���                                    Bxq��:  
�          @�(�@qG����@!�A�  C�=q@qG��ڏ\�n{�أ�C�w
                                    Bxq���  T          @�ff@�\)���H@AG�A�{C��q@�\)��=q�aG�����C�#�                                    Bxq���  �          @��R@�{���
@>�RA�Q�C��\@�{�ҏ\�����{C�                                    Bxq��,  "          @�p�@��R���H@j�HA�
=C�~�@��R��p�>�@^�RC�#�                                    Bxq���  �          @��@�  ��(�@i��A�  C��
@�  ��
=?�@�p�C�g�                                    Bxq��x  �          @�p�@������H@�=qA�=qC��@�������?u@߮C���                                    Bxq��  �          @��@�������@�33Bp�C��@�����=q@33At  C�W
                                    Bxq��  "          @�33@W����
@�  A��HC���@W���\?(��@���C��R                                    Bxq�j  �          @�@L����(�@Z=qA�z�C�T{@L����\)���Ϳ@  C��                                    Bxq�"  
�          @���@O\)��ff@XQ�AΏ\C���@O\)��녽#�
��\)C�P�                                    Bxq�0�  �          @���@J�H�ə�@S�
A��C�Z�@J�H��������C��                                    Bxq�?\  "          @���@J=q�ə�@L��A�\)C�P�@J=q��녾u���
C��                                    Bxq�N  "          @�
=@p��У�@Q�A�C�S3@p���G���=q� ��C�N                                    Bxq�\�  �          @���@#�
��  @W�A�=qC���@#�
��=q�8Q쿥�C��f                                    Bxq�kN  �          @�(�@0  ��z�@l(�A�\)C���@0  ��(�>#�
?���C�8R                                    Bxq�y�  "          @�(�@8Q���\)@w�A�33C�h�@8Q���=q>���@8Q�C��q                                    Bxq���  
�          @���@>�R����@��
A���C�` @>�R���?G�@�  C�H�                                    Bxq��@  �          @�{@/\)���@�\)B��C�f@/\)��=q?��RAz�C���                                    Bxq���  
�          @�33@L(����H@�=qB33C�k�@L(���Q�?�p�A{C��                                     Bxq���            @��@aG���(�@���B=qC��f@aG����H?�Q�A3�C���                                    Bxq��2  
�          @�33@H����ff@�p�B(�C�.@H����
=?\A>�RC���                                    Bxq���  
�          @�@\)���@uA��C���@\)����?��\@�C�|)                                    Bxq��~  
�          @���@�(����\@Z�HA�Q�C���@�(����H?z�@�\)C�3                                    Bxq��$  
Z          @���@����@[�A�Q�C�` @����?
=q@�p�C��3                                    Bxq���  �          @�(�@g
=��p�@g
=A��C�<)@g
=�ָR>��@c�
C�#�                                    Bxq�p  �          @�\@�z���=q@B�\A�  C��f@�z����
>\)?��C��{                                    Bxq�  �          @�G�@�z����R@6ffA�Q�C�G�@�z���
=>��?�C�>�                                    Bxq�)�  
�          @�G�@�������@S�
AиRC��@�������?E�@�(�C�)                                    Bxq�8b  �          @��H@�����  @[�A�p�C�!H@�������?&ff@�p�C��                                     Bxq�G  �          @��
@������@P��AʸRC��R@����\>�(�@Mp�C�ff                                    Bxq�U�  �          @�33@�����\@HQ�A�z�C�Ф@����p�>k�?�G�C�Ǯ                                    Bxq�dT  T          @�(�@s33���
@C33A�z�C�s3@s33���
��Q�(��C��                                    Bxq�r�  "          @�\)@J=q��z�@
=A�=qC��
@J=q��
=������C�0�                                    Bxq���  "          @��@X������@ ��A�
=C��\@X���ڏ\�L����Q�C�/\                                    Bxq��F  
�          @��@[���ff@eA�RC���@[��ָR>�@i��C���                                    Bxq���  �          @�=q@mp���Q�@h��A�C��q@mp��ҏ\?�R@�ffC���                                    Bxq���  T          @�
=@�=q��=q@�A�ffC��f@�=q���ÿ����\C�Ф                                    Bxq��8  "          @�
=@�z���(�@:=qA��RC�u�@�z���z�>#�
?��C���                                    Bxq���  "          @�@�p���
=@,��A�
=C���@�p�����=u>��C��                                    Bxq�ل  T          @陚@�������@.{A���C��@�����\)>.{?��C��                                    Bxq��*  
�          @��
@�Q����
@Y��A�(�C��q@�Q���ff?xQ�@�33C���                                    Bxq���  �          @��@�ff���\@}p�B�HC�AH@�ff����?�{A)C���                                    Bxq�v  �          @�@�G���
=@P��A��HC�@ @�G��˅>���@!�C�<)                                    Bxq�  T          @��
@l(��Å@R�\A��C��3@l(���{=u>�C��                                    Bxq�"�  "          @�  @�������@�\)B
=C�G�@�����{?�33A&�HC�                                      Bxq�1h  �          @�\@�����@��B \)C��{@����33@�HA��C��                                    Bxq�@  
�          @���@�����33@��\B

=C�(�@������?�A[�
C��                                    Bxq�N�  �          @��@�Q��l��@�ffB!G�C�P�@�Q����\@*�HA�\)C�|)                                    Bxq�]Z  �          @�z�@�=q�I��@�\)B�\C�3@�=q���@.�RA�(�C��                                     Bxq�l   �          @�ff@�  ��  @���A�p�C��{@�  ��z�?У�AB�\C���                                    Bxq�z�  �          @�@�����G�@���B	��C��)@�����  ?�Q�ANffC���                                    Bxq��L  
�          @�@y�����@��HB G�C�O\@y����G�?��HA��C�e                                    Bxq���  �          @�Q�@qG���=q@�{Bz�C�T{@qG���G�?�{A@(�C���                                    Bxq���  �          @��H@�\)���@7
=A���C��R@�\)����=�Q�?333C���                                    Bxq��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq�Ҋ  
?          @�33@��
���
?�G�AO�C���@��
���׿����G�C�w
                                    Bxq��0  "          @�
=@�G���?�ffA2�HC��@�G���ff���H�(��C��R                                    Bxq���  �          A�@y����?�
=A\)C��\@y����ff���t��C�R                                    Bxq��|  �          A ��@~�R���H?��AQ�C��@~�R��p���p��dz�C�h�                                    Bxq�"  "          A ��@��H���?�=q@�\)C�}q@��H�љ��	���x��C��)                                    Bxq��  �          A@l����p�?h��@�ffC��@l����G��p�����C�G�                                    Bxq�*n  
�          @�
=@�Q����?�  @�
=C�B�@�Q��У��p����RC��\                                    Bxq�9  T          @��H@��H��z�?Q�@��C�Ф@��H��G���\���C��f                                    Bxq�G�  �          @��\@�����>��@[�C�ٚ@�����
=�����=qC��\                                    Bxq�V`  �          @�z�@�Q���ff<�>k�C�S3@�Q���������p�C��                                    Bxq�e  �          @�p�@�33��  �����z�C��@�33�n{��
�z�\C��3                                    Bxq�s�  �          @�\@�\)���ÿ�{�&�RC��\@�\)�>{�B�\��{C�=q                                    Bxq��R  
�          @�@�(�����Q�&ffC�Z�@�(���ff�>�R��C��R                                    Bxq���  �          @���@z=q����p��z�C��=@z=q����  ����C���                                    Bxq���  T          @���@�{������
�S�
C��3@�{��33�����ffC���                                    Bxq��D  "          @���@��\��p������z�C��R@��\�[����
���C��q                                    Bxq���  
(          @�@�=q��������z�C���@�=q�Y����{�  C��                                    Bxq�ː  
�          @�\)@��������
��\)C�H@����a�����	��C�'�                                    Bxq��6  �          @�p�@�����ÿ����l(�C�&f@���vff����33C��                                     Bxq���  T          @�@����p��@�����\C��@���C�
��z��z�C�n                                    Bxq���  �          @�p�@�
=��Q�k���Q�C���@�
=��G��K��Ώ\C��{                                    Bxq�(  T          @���@�{��G���z����C�%@�{���\�.�R��z�C��                                    Bxq��  �          @���@�33��ff��  �b�RC�4{@�33�i���qG�����C���                                    Bxq�#t  "          @��@��
��G���G��"=qC��3@��
��p��^�R��C��3                                    Bxq�2  �          @��H@�z����H��G��E��C��
@�z�����n�R��z�C�q�                                    Bxq�@�  �          @�\@�p���{���\z�C�7
@�p��y���s33�33C�C�                                    Bxq�Of  T          @��@�  ���\�XQ���{C��\@�  �Q����R�,  C��                                    Bxq�^  �          @�{@�ff��G��#�
���C�5�@�ff�J=q��p���HC�f                                    Bxq�l�  T          @�33@�����ÿ�z��zffC�9�@���N{�n{��
=C��                                    Bxq�{X  T          @���@�(�����Q�����C���@�(�����HQ����C�P�                                    Bxq���  
�          @�=q@s�
���H�\)��
=C�(�@s�
�����>{��G�C�T{                                    Bxq���  T          @�p�@p����  ��\)�
=C���@p����G��2�\��p�C�XR                                    Bxq��J  "          @�
=@���G��fff��C��R@�����H����=qC���                                    Bxq���  "          @޸R@�  ��녿����3�C�W
@�  ���c33��  C��                                     Bxq�Ė  �          @�z�@fff�\�B�\��p�C�K�@fff��(��Vff��G�C���                                    Bxq��<  
�          @�R@|�����R�J=q��=qC��H@|����Q��Tz���(�C�q                                    Bxq���  �          @�{@x����{��\)��C���@x����33�g
=��
=C�\)                                    Bxq���  �          @�z�@j�H���׿�����C��@j�H��p��h����\)C�Y�                                    Bxq��.  �          @���@e��ff��z��=��C�)@e��Q��p���ffC�C�                                    Bxq��  �          @ٙ�@q����R�����7�C�Y�@q����\�e� 
=C���                                    Bxq�z  �          @��H@e���ff���G�C��@e����
�a�����C���                                    Bxq�+   �          @�\)@����Q�z�H�ffC��{@������J�H��33C���                                    Bxq�9�  "          @�G�@�����{�p����
=C�S3@��������?\)��\)C�0�                                    Bxq�Hl  �          @أ�@����33�J=q��{C���@�������3�
��=qC��
                                    Bxq�W  T          @�Q�@�p���녿#�
����C���@�p���G��0����33C��\                                    Bxq�e�  
�          @�Q�@�Q���p����R�)��C�>�@�Q���  �'�����C��                                    Bxq�t^  �          @��@�����G�=#�
>�Q�C��@�����G��\)���
C��                                    Bxq��  �          @У�@�ff��
=>k�@   C�
@�ff���\��Q���ffC�=q                                    Bxq���  �          @�Q�@�
=��
==�\)?��C���@�
=��G���(���
=C�3                                    Bxq��P  �          @�Q�@�{��  �W
=��{C���@�{��ff��R��Q�C�O\                                    Bxq���  �          @�  @�����������C�j=@������H������C���                                    Bxq���  �          @�  @�  ��=q�����C��)@�  �j�H�ff���C�*=                                    Bxq��B  !          @У�@��H�/\)?��A�Q�C��@��H�Q�?�@�(�C���                                    Bxq���  �          @���@��\�p��?���AG\)C��@��\�~�R�\�`  C�B�                                    Bxq��  T          @ȣ�@����?O\)@�RC���@���G���\)�$��C���                                    Bxq��4  T          @ə�@�\)��(�?333@��
C�5�@�\)�������C�C���                                    Bxq��  �          @��@H����ff>��@z�C��q@H����G�����RC���                                    Bxq��  
�          @�p�@=p���{>��@�
C��
@=p���Q�������C���                                    Bxq�$&  �          @θR@0�����H=�Q�?W
=C���@0�����\������C���                                    Bxq�2�  �          @˅@'���G����
�5C�1�@'����R�!G����HC�]q                                    Bxq�Ar  �          @��
@.{��  ��p��U�C���@.{����0����C�"�                                    Bxq�P  �          @�z�@$z����ÿ@  �ٙ�C���@$z���p��G����HC��f                                    Bxq�^�  
�          @θR@   ��ff�L�;�C�j=@   ���
�#33��ffC�~�                                    Bxq�md            @�  @+���p��L�Ϳ�C�+�@+������*�H��z�C�o\                                    Bxq�|
  8          @�z�@G���=q��Q�B�\C��@G�������H���
C��3                                    Bxq�  T          @�z�@N{��  >���@,��C�.@N{���
�����RC�
                                    BxqV  �          @�p�@Tz����>��?�{C���@Tz������
=q��\)C��\                                    Bxq§�  "          @�z�@Z�H����>#�
?�
=C�0�@Z�H��
=�
=��Q�C�G�                                    Bxq¶�  p          @�33@Z=q��=q?!G�@�\)C�XR@Z=q��33����p  C��                                    Bxq��H  
>          @��@n{��(�?n{A�C��
@n{��������:{C�,�                                    Bxq���  �          @˅@n�R���?�G�A6�\C�j=@n�R��녿n{��HC�8R                                    Bxq��  �          @��@`������?�(�AW33C�z�@`�����@  ���
C��                                    Bxq��:  
�          @��@S�
���?У�Ap(�C�y�@S�
���\�!G�����C��\                                    Bxq���  	�          @ʏ\@H����{?�ffA�ffC���@H����
=��\���C���                                    Bxq��  T          @��H@HQ�����?�(�A���C���@HQ���Q쾮{�A�C���                                    Bxq�,  
�          @�z�@<�����R@�RA�C���@<���������ffC��)                                    Bxq�+�  
�          @�z�@Fff���@	��A�
=C���@Fff���\�.{���C��f                                    Bxq�:x  �          @�(�@E��(�@(�A���C��@E��=q��G���  C���                                    Bxq�I  �          @�(�@B�\��{@$z�A�\)C��\@B�\���>��R@5C�U�                                    Bxq�W�  
�          @�{@*�H���
@P  A�ffC�^�@*�H����?uA	p�C�j=                                    Bxq�fj  T          @���@Mp����R@p�A��HC�u�@Mp���G�>W
=?�33C��                                    Bxq�u  
�          @���@Fff����@��A�=qC��f@Fff���H=�?��C���                                    BxqÃ�  T          @��@G
=��33@ ��A�(�C�W
@G
=��ff>��R@3�
C��)                                    BxqÒ\  
�          @��H@I����  @.{A�ffC���@I����ff?��@�  C��                                    Bxqá  "          @�G�@N{���
@333A��C�n@N{���?+�@��
C��                                     Bxqï�  T          @���@Q�����@�A��C�&f@Q���33>W
=?���C��)                                    BxqþN  
�          @�\)@J�H��p�@z�A��C�e@J�H���H��G���G�C�Z�                                    Bxq���  
�          @ƸR@Vff���@
=A���C���@Vff��
=>�=q@\)C�U�                                    Bxq�ۚ  
�          @Ǯ@\����p�@G�A��\C�B�@\����>8Q�?ٙ�C��
                                    Bxq��@  
�          @�Q�@a���=q@��A�ffC���@a�����>�33@K�C�5�                                    Bxq���  �          @�
=@p����=q?�
=A���C��)@p�����R��\)�.{C��
                                    Bxq��  
�          @�
=@z�H��z�?��
A>�\C��@z�H��Q�0����C��)                                    Bxq�2  "          @���@p�����\@�A���C�|)@p�����H>��@�C��=                                    Bxq�$�  
�          @�z�@qG����@$z�A�C�@ @qG�����?333@��C�3                                    Bxq�3~  
�          @��H@r�\�|(�@*=qA�\)C��\@r�\��?\(�Ap�C���                                    Bxq�B$  �          @�=q@y���s�
@*=qA�{C���@y�����?k�A
�RC�H�                                    Bxq�P�  
�          @���@xQ��p��@.�RA֣�C��@xQ���G�?�  A��C�AH                                    Bxq�_p  	�          @�Q�@����g�@'
=A�C�H@������?s33A�C�Y�                                    Bxq�n  
�          @��@�G��g�@!�A��
C��@�G���=q?aG�A
=C���                                    Bxq�|�  �          @�  @����aG�@.{Aי�C�g�@������?��A(Q�C���                                    Bxqċb  �          @���@��H�Vff@8Q�A��C�]q@��H���R?���AJ{C�{                                    BxqĚ  
�          @�Q�@�p��<��@K�A���C�Y�@�p��~{?�  A��HC�5�                                    BxqĨ�  �          @�  @��\�I��@C�
A��C�*=@��\��33?���ArffC�j=                                    BxqķT  �          @���@�  �R�\@Dz�A���C�H�@�  ���?\Aip�C���                                    Bxq���  
�          @���@r�\�S33@S33BG�C�xR@r�\���\?�p�A�{C��3                                    Bxq�Ԡ  
Z          @��@�G��dz�@.�RA���C�>�@�G����?�{A(��C�ff                                    Bxq��F  "          @��@����o\)@#33A�
=C���@�����{?^�RA��C�
                                    Bxq���  T          @���@z�H�e�@7
=A�\C���@z�H��p�?�p�A;�
C���                                    Bxq� �  
�          @��@��H�n�R@\)A��\C�Ф@��H���?��@��C��q                                    Bxq�8  
�          @�{@���g
=@33A�z�C�Y�@����
=?5@��HC�)                                    Bxq��  �          @���@|���Y��@1�A�z�C��@|�����R?�p�A@z�C��3                                    Bxq�,�  �          @�z�@}p��Q�@7
=A��C�0�@}p���(�?�{AUG�C��                                    Bxq�;*            @�{@w
=�N{@I��B 33C�R@w
=��{?�33A�
=C�U�                                    Bxq�I�  8          @�p�@g��X��@N{B��C�o\@g����
?�33A��C���                                    Bxq�Xv  �          @��R@fff�Mp�@_\)B�C�(�@fff���?�p�A��C���                                    Bxq�g  �          @�@h���QG�@W
=B	�HC�f@h�����?�=qA�C��                                    Bxq�u�  �          @��
@hQ��P  @P��BG�C��@hQ���Q�?�G�A�\)C�<)                                    Bxqńh  �          @���@\���R�\@^{B(�C�/\@\�����
?�
=A�
=C�"�                                    Bxqœ  
(          @��
@a��L(�@[�B�C��@a�����?�Q�A�{C��=                                    Bxqš�  �          @�33@Fff�hQ�@UB  C�E@Fff��(�?�Q�A���C��                                    BxqŰZ  T          @���@c�
�hQ�@-p�A�(�C�5�@c�
��z�?�{A0��C���                                    Bxqſ   
�          @��\@u��K�@@��A���C�&f@u����H?ǮAx��C���                                    Bxq�ͦ  �          @�33@n{�E�@QG�BQ�C�(�@n{��33?�A��C��                                    Bxq��L  
�          @���@����Fff@.{A�C�1�@����y��?�=qAU�C��                                    Bxq���  �          @���@|(��C�
@8Q�A�(�C��@|(��{�?�  Aqp�C���                                    Bxq���  T          @���@mp��+�@`  Bz�C��@mp��s�
@p�A��HC�%                                    Bxq�>  �          @�  @p  �1G�@W
=B�C��@p  �u�@�
A��
C�0�                                    Bxq��  
(          @�  @|(��R�\@$z�A���C�R@|(�����?���A6{C�=q                                    Bxq�%�  "          @�G�@|���G�@3�
A陚C��{@|���}p�?�AdQ�C�y�                                    Bxq�40  
�          @�=q@u�P��@7�A�33C���@u��33?�Ac33C���                                    Bxq�B�  T          @�=q@qG��K�@E�B   C��@qG����?�z�A�Q�C�H�                                    Bxq�Q|  �          @���@n{�@  @QG�B	��C��3@n{��Q�?�33A�\)C�o\                                    Bxq�`"  �          @���@aG��2�\@h��B33C��\@aG��|��@z�A��
C��R                                    Bxq�n�  
�          @�  @]p��AG�@\��B�C�o\@]p���33@z�A��\C�R                                    Bxq�}n  "          @�  @dz��G
=@P  B	��C�~�@dz���33?���A�(�C���                                    Bxqƌ  �          @�G�@e��O\)@K�BffC��3@e���{?޸RA��
C�C�                                    Bxqƚ�  �          @�=q@b�\�H��@Y��B�RC�/\@b�\��?�p�A��HC��                                    BxqƩ`  T          @��H@i���:�H@`��B�C���@i������@
�HA�  C��                                    BxqƸ  T          @�=q@n�R�5�@]p�B�HC�l�@n�R�z=q@
=qA�=qC��3                                    Bxq�Ƭ  p          @��H@l���AG�@VffB(�C�j=@l����G�?��RA��C�<)                                    Bxq��R  
�          @��H@`���P��@U�BG�C��@`����Q�?��A��HC���                                    Bxq���  �          @�=q@e��L��@Q�B	Q�C�
@e���{?�{A�\)C�AH                                    Bxq��  
�          @��\@mp��G�@N�RB��C���@mp����H?���A�Q�C�q                                    Bxq�D  
�          @���@q��:�H@R�\B
ffC�%@q��{�?�p�A�{C��\                                    Bxq��            @��\@u��A�@L(�B�\C���@u���  ?���A�C���                                    Bxq��  
p          @�(�@u��J=q@H��B{C�<)@u����H?�G�A�
=C��                                    Bxq�-6  
Z          @�@r�\�Q�@J�HBQ�C���@r�\���R?�  A�
=C��R                                    Bxq�;�  �          @�(�@l���Y��@C33A��
C���@l������?˅A{
=C�h�                                    Bxq�J�  "          @�z�@����Mp�@7�A�33C���@�����G�?�  Ak�C�t{                                    Bxq�Y(  "          @���@j=q�b�\@?\)A�33C���@j=q��(�?��RAiC���                                    Bxq�g�  T          @�ff@n{�aG�@A�A���C�G�@n{��(�?��Ao\)C�(�                                    Bxq�vt  
�          @�p�@�p��@  @:�HA�G�C��@�p��w
=?У�A�=qC���                                    Bxqǅ  �          @�z�@�
=�5�@>�RA�G�C�{@�
=�n{?�  A�ffC�N                                    BxqǓ�  �          @�(�@��1�@C�
A�33C�*=@��l��?���A�z�C�8R                                    BxqǢf  T          @�33@�Q��5@333A�C�"�@�Q��j�H?�=qA|  C��                                    BxqǱ  
�          @�33@���<��@.�RA�33C���@���o\)?�p�Ak\)C�G�                                    Bxqǿ�  
v          @��@�G��QG�@,(�A�z�C��@�G�����?�=qAQ��C��R                                    Bxq��X  �          @��\@�G��?\)@$z�A�
=C��@�G��n{?���AQ�C��\                                    Bxq���  
�          @�=q@�
=�2�\@{Aȏ\C���@�
=�_\)?�ffAN=qC���                                    Bxq��  "          @���@�(��0��@$z�A�33C��@�(��_\)?�z�Ab{C��{                                    Bxq��J  
Z          @�Q�@��\�333@#�
A��HC��f@��\�a�?���A^ffC�aH                                    Bxq��  
�          @�{@�  �%�@0  A�C�Y�@�  �X��?�33A��HC��\                                    Bxq��  "          @�{@����{@333A�ffC��@����S33?޸RA�z�C�.                                    Bxq�&<            @�  @����E�@*�HA�(�C�L�@����u�?�33Ab{C�:�                                    Bxq�4�  T          @�G�@vff�P��@4z�A�C��f@vff����?�p�Amp�C���                                    Bxq�C�  
�          @�=q@z�H�S�
@0��A�z�C���@z�H��=q?�z�A`(�C��{                                    Bxq�R.  
Z          @��H@w
=�S�
@6ffA�Q�C��
@w
=��33?�  An{C���                                    Bxq�`�  
v          @��H@r�\�Y��@7�A�\)C��@r�\��ff?�p�Ak\)C�f                                    Bxq�oz            @�=q@�G��L(�@-p�A�Q�C��R@�G��|(�?�z�A`��C�ٚ                                    Bxq�~             @�=q@�  �<(�@*�HA�Q�C��f@�  �l(�?��HAh��C��H                                    BxqȌ�            @�G�@����HQ�@*�HA��
C�%@����w�?��A_\)C�(�                                    Bxqțl  T          @���@�=q�H��@'�A��
C�33@�=q�w
=?���AX��C�H�                                    BxqȪ  
�          @�G�@u��W
=@.�RA��C�\)@u���33?���A\z�C���                                    Bxqȸ�  �          @��@z�H�Tz�@,��A��C�ٚ@z�H���?�\)AZffC���                                    Bxq��^  �          @�=q@g
=�g
=@2�\A�C���@g
=��33?�{AXQ�C��{                                    Bxq��  �          @��@qG��XQ�@5�A�33C��@qG�����?�(�Ak
=C�                                      Bxq��  T          @���@s�
�Z�H@(Q�AٮC��@s�
���
?��
AM�C�XR                                    Bxq��P  �          @�G�@u��R�\@3�
A��HC���@u����?��RAo�C��f                                    Bxq��  �          @���@n�R�\��@3�
A�p�C��
@n�R���R?�Q�Aep�C��                                     Bxq��  �          @��\@tz��L��@@��A�
=C��)@tz�����?�(�A��C���                                    Bxq�B  "          @�=q@tz��a�@(Q�A֣�C�� @tz���
=?�  AF=qC��                                    Bxq�-�  
Z          @��@s33�e�@!G�A�(�C�XR@s33��\)?���A4Q�C��
                                    Bxq�<�  
�          @�33@g��e@1G�A�{C���@g����\?�{AYC��{                                    Bxq�K4  T          @�=q@_\)���
@
=qA�(�C�)@_\)���?(�@���C��=                                    Bxq�Y�  
�          @��H@k��z=q@�A��C��f@k���ff?L��@���C���                                    Bxq�h�  �          @���@i����=q@  A�p�C���@i�����H?:�H@�=qC�9�                                    Bxq�w&  T          @�(�@y���p  @�A��C��@y������?^�RA�
C�R                                    BxqɅ�  �          @�z�@}p��j�H@z�A�  C��
@}p����?n{A��C��                                    Bxqɔr  �          @�z�@|(��y��?�(�A�=qC���@|(����\?�@�33C�#�                                    Bxqɣ  �          @��@vff�z=q@�\A�G�C�J=@vff���
?
=@�=qC���                                    Bxqɱ�  �          @�(�@qG����?��HA���C�o\@qG���\)>�@���C�f                                    Bxq��d  
�          @��@u��u@z�A�\)C�p�@u�����?aG�A	�C��H                                    Bxq��
  T          @�{@~{�^�R@*�HA�
=C�\)@~{��?��AQC���                                    Bxq�ݰ  �          @��@�
=�?\)@7�A��
C�\)@�
=�q�?�Q�A��HC�
                                    Bxq��V  T          @���@z�H�L(�@@  A���C�s3@z�H��Q�?�  A���C�,�                                    Bxq���  
�          @���@���`��@�A���C���@�����
?�{A-C�E                                    Bxq�	�  �          @�(�@���e@  A�ffC�Ff@����(�?k�A�
C�@                                     Bxq�H  "          @��@�Q��W�@�
A��C��{@�Q��|��?��A#
=C��R                                    Bxq�&�  �          @�@�(��]p�?��RA�\)C��)@�(��z�H?8Q�@���C��                                    Bxq�5�  
�          @�p�@�G��Z=q@�RA��C��H@�G��}p�?xQ�A{C��                                    Bxq�D:  
�          @��@��j=q?�p�A��C�` @����?&ff@ʏ\C��R                                    Bxq�R�  "          @��@�33�|��?�=qAx(�C��@�33��  >8Q�?�p�C��)                                    Bxq�a�  T          @��
@���x��?��Ar�RC�L�@����>#�
?�ffC�Ff                                    Bxq�p,  
Z          @���@�(��xQ�?��A��C�e@�(����R>��@#�
C�@                                     Bxq�~�  �          @�z�@�����H?��\A\)C��
@����{�Ǯ�r�\C�8R                                    Bxqʍx  T          @���@u��33?s33A��C���@u��p��
=q��Q�C�|)                                    Bxqʜ  �          @�z�@�
=�u?�\)AX��C��{@�
=���\<�>��RC��R                                    Bxqʪ�  �          @�p�@~{�~{?��A�G�C�|)@~{���H>\@n{C�8R                                    Bxqʹj  �          @�@}p�����?��
A�(�C�C�@}p���z�>�Q�@aG�C��                                    Bxq��  �          @�p�@��H�n�R?��HAd��C���@��H��  >#�
?�G�C���                                    Bxq�ֶ  
�          @�z�@�
=�s�
?�(�Ag�C��{@�
=���\>\)?���C���                                    Bxq��\  �          @�z�@�(��R�\@��A��RC��\@�(��s33?s33A��C���                                    Bxq��  "          @��@��H�QG�@A�{C���@��H�qG�?h��A�C���                                    Bxq��  "          @��@���1G�@=qA�33C�~�@���Z=q?���AX��C��f                                    Bxq�N  
�          @�{@�p��1G�@��A�
=C���@�p��Y��?�\)AV=qC���                                    Bxq��  "          @�
=@�33�1G�@%�A�\)C�xR@�33�]p�?�ffApQ�C���                                    Bxq�.�  
�          @�
=@��R�8Q�@*�HA�\)C���@��R�e�?˅Ax��C��                                     Bxq�=@  
(          @�{@�����H@<��A�G�C���@����P  @ ��A���C�*=                                    Bxq�K�  "          @��R@�Q��,(�@A�
=C�K�@�Q��S33?���AQG�C��=                                    Bxq�Z�  "          @�
=@���P��@
=A�ffC���@���u?�Q�A7�C�t{                                    Bxq�i2  
�          @�ff@���E?��A�Q�C�s3@���`  ?333@�G�C��f                                    Bxq�w�  �          @�ff@���I��@ ��A�=qC�\@���r�\?�\)AUG�C�}q                                    Bxqˆ~  T          @�{@�(��AG�@%A��C��{@�(��l(�?��RAi��C��\                                    Bxq˕$  
Z          @�p�@�(��xQ�?��A�\)C�g�@�(���ff>��
@FffC�C�                                    Bxqˣ�  "          @��@�=q��(�?��A%�C�J=@�=q��  ���
�Dz�C��                                     Bxq˲p  T          @�z�@�=q��p�?E�@�=qC�'�@�=q��ff�
=����C�                                    Bxq��  
Z          @���@�z����\?�G�A�C��
@�z���{��{�P��C�T{                                    Bxq�ϼ  
�          @�(�@�������?�=qA)C���@������;�  ��HC�w
                                    Bxq��b  "          @��\@��H�xQ�?�p�Ak�C�E@��H����>8Q�?�G�C�N                                    Bxq��  T          @��\@����|(�?\Aq�C��=@�����
=>B�\?�33C���                                    Bxq���  �          @��H@����z�H?���Ag33C��R@�����>\)?�33C�                                    Bxq�
T  
�          @�=q@���]p�?+�@�33C���@���_\)����  C�o\                                    Bxq��  
�          @�=q@���u�?&ff@�C�1�@���u�
=����C�'�                                    Bxq�'�  T          @�(�@����q�?�  AEC�\)@����~�R<#�
=�\)C��q                                    Bxq�6F  "          @�(�@��H�hQ�?�(�Ah��C��@��H�z=q>��@!�C�                                      Bxq�D�  
�          @�(�@�ff�p��?�
=A�
=C��@�ff��33>�(�@�z�C��                                     Bxq�S�  
�          @��@�ff�x��?�=qAQ�C��
@�ff���=L��>�C��                                    Bxq�b8  �          @��H@�=q�c�
?�=qAU��C�@ @�=q�s33>��?��
C�XR                                    Bxq�p�  
�          @��@�(��^{?�\)A���C��\@�(��s33>�@��\C��                                    Bxq��  
�          @��\@�Q��[�?���A���C���@�Q��w�?J=q@���C��                                    Bxq̎*  
�          @�(�@����[�@�A���C�=q@����~�R?�33A4��C�q                                    Bxq̜�  �          @��@���r�\?!G�@ǮC�Z�@���r�\����ffC�S3                                    Bxq̫v  "          @��H@�G��p  ?��HA@  C�n@�G��|(�    �#�
C��R                                    Bxq̺  �          @�33@�=q�x��?���Axz�C�  @�=q��{>�z�@5C��                                    Bxq���  
�          @��\@{��{�?�z�A�G�C��@{���  >\@mp�C�e                                    Bxq��h  �          @�33@���qG�?�(�A��C���@�����
>�@�
=C�y�                                    Bxq��  �          @��@�\)�r�\?У�A�(�C��@�\)���>Ǯ@r�\C���                                    Bxq���  �          @�p�@�����?���A&�HC�ٚ@����{�aG��
�HC�c�                                    Bxq�Z  �          @��@�p����?�G�Ap�C��@�p���p���\)�,(�C�~�                                    Bxq�   
Z          @�p�@\)���?Q�@��RC�aH@\)��33������C�<)                                    Bxq� �  �          @��\@]p����H?k�AG�C��H@]p����Ϳ���z�C�Q�                                    Bxq�/L  
�          @��@tz���\)?���AQp�C��@tz������
�aG�C�^�                                    Bxq�=�  �          @���@x����?�z�A]p�C�xR@x�����=�\)?8Q�C��{                                    Bxq�L�  T          @���@�33�vff?}p�A�C�B�@�33�~{�aG��
=C��3                                    Bxq�[>  
(          @���@�
=��Q�?\(�AC�8R@�
=���\�����w�C��R                                    Bxq�i�  	�          @���@�(����H?fffA��C���@�(���p���p��g
=C�]q                                    Bxq�x�  �          @��
@�  �e?�33A�=qC��f@�  �z�H>��H@��
C��f                                    Bxq͇0  T          @��@����`��?��
A��\C�L�@����xQ�?#�
@ʏ\C��                                     Bxq͕�  "          @�ff@��
�Z=q@#33A��C�.@��
����?�z�A\  C�ٚ                                    Bxqͤ|  �          @��@��
�]p�?���A�(�C��
@��
�xQ�?Q�A Q�C�4{                                    Bxqͳ"  T          @��@���[�?��RA��C��H@���w�?^�RA�C�1�                                    Bxq���  "          @�z�@�33�Z=q@   A�33C��@�33�w
=?c�
A�C�9�                                    Bxq��n  �          @�@��E?�z�A��\C�H�@��`��?fffAz�C���                                    Bxq��  "          @��@�p��>�R?�(�A�ffC���@�p��[�?}p�A\)C�ٚ                                    Bxq���  �          @��
@���=p�?��HA�Q�C��f@���Z=q?}p�A�
C��                                    Bxq��`  �          @�(�@�Q��:�H?���A��\C�=q@�Q��U�?aG�A��C��                                    Bxq�  �          @��H@��(�?��A��C��3@��8��?�=qA+�C���                                    Bxq��  �          @�33@���&ff?�=qA�  C��@���AG�?z�HA�\C�,�                                    Bxq�(R  T          @�33@��H�0  ?�ffA�
=C�7
@��H�I��?h��A�\C�t{                                    Bxq�6�  T          @��\@��
�3�
@	��A�G�C�U�@��
�Tz�?�(�ABffC�&f                                    Bxq�E�  
�          @��\@�(��+�@0  AᙚC�E@�(��W�?�A���C�5�                                    Bxq�TD  "          @��@��,��@,��A�ffC�Q�@��W�?��A��HC�Y�                                    Bxq�b�  "          @�33@�33�,(�@3�
A�z�C�*=@�33�X��?�33A�=qC�
=                                    Bxq�q�  �          @��H@��
�XQ�@(�A���C�L�@��
�xQ�?�{A1�C�^�                                    Bxq΀6  �          @��H@������H?xQ�A��C�J=@�����ff��=q�)��C��                                    BxqΎ�  
�          @��@|(��z=q?�ffA�C�� @|(���Q�?z�@�\)C�aH                                    BxqΝ�  "          @�33@z�H�j�H@�A��
C�u�@z�H��p�?�{A/�
C���                                    Bxqά(  �          @�(�@�Q��n�R@�A�p�C��@�Q���?s33A��C��                                    Bxqκ�  "          @�p�@����g
=@��A��C�3@�������?�  AC\)C�3                                    Bxq��t  	�          @�z�@s33�g�@'�A�Q�C�8R@s33��
=?��HAf�HC��                                    Bxq��  
�          @��@�33�8��@7�A�C�k�@�33�e?�z�A�(�C�c�                                    Bxq���  
�          @���@����
@-p�A�Q�C��@���G�@33A��C�&f                                    Bxq��f  
�          @�(�@�=q�=p�@
=A��C��@�=q���@33A��C��                                    Bxq�  "          @���@��ÿ��
?�33A��RC���@��ÿ��H?��
AJffC���                                    Bxq��  �          @�p�@�녿�(�@��A��
C�@ @����?У�A�C��\                                    Bxq�!X  �          @�@��R�z�@�A�ffC���@��R�+�?�p�A��\C��\                                    Bxq�/�  "          @��
@���!�@(�A��C�!H@���C33?�\)AX��C��\                                    Bxq�>�  
�          @��\@�\)�Q�@��A\C���@�\)�>{?�{A��HC��3                                    Bxq�MJ  "          @�33@�Q��(�@A�G�C�u�@�Q��@  ?�ffAvffC��                                    Bxq�[�  T          @���@�����@"�\A��C���@����*�H?�{A��C��
                                    Bxq�j�  T          @���@����'
=@p�A���C��3@����HQ�?���AX��C�g�                                    Bxq�y<  �          @�(�@�p��!�?�(�A��C�h�@�p��>�R?�A8��C�b�                                    Bxqχ�  �          @��@���8��?�Q�A��C��q@���P  ?L��@���C��                                    Bxqϖ�  T          @��@��R�L��?�=qAw�C��H@��R�aG�?��@�(�C��)                                    Bxqϥ.  T          @�z�@���I��?�G�Am�C�+�@���\��?\)@�{C��{                                    Bxqϳ�  �          @�33@����E?�{AV�HC���@����U>�(�@��C��                                     Bxq��z  "          @��H@���`  ?z�HA\)C�B�@���hQ�#�
����C���                                    Bxq��   �          @���@���s�
?�{A[
=C�g�@������>k�@33C��3                                    Bxq���  �          @�
=@~{��G�?�@���C�9�@~{���ÿ!G����
C�G�                                    Bxq��l  �          @��@�G��n�R?(�@�p�C��H@�G��p  ����=qC�k�                                    Bxq��  �          @�\)@�{�`  ?^�RAQ�C��@�{�fff�\)��z�C�~�                                    Bxq��  	�          @���@�33�S�
?�z�A9p�C�%@�33�`  >B�\?��C�]q                                    Bxq�^  "          @���@�Q��`  ?Q�A=qC��@�Q��e�B�\��z�C��H                                    Bxq�)  �          @���@����c�
>�Q�@b�\C��=@����aG��&ff��
=C�{                                    Bxq�7�  T          @�=q@�=q�w�>���@P��C��@�=q�s�
�B�\��Q�C�O\                                    Bxq�FP  
�          @��H@��
�w�=��
?E�C�@ @��
�o\)��  �C��q                                    Bxq�T�  "          @�=q@��H�vff>��?��RC�/\@��H�o\)�k���\C��R                                    Bxq�c�  "          @��@�33�`  ��ff���C�^�@�33�N�R��33�`z�C�o\                                    Bxq�rB  �          @��\@�p��B�\�5��  C�'�@�p��.{�\�r=qC��                                    BxqЀ�  T          @�z�@��H�-p������0��C��)@��H�33��=q��G�C�޸                                    BxqЏ�  "          @�
=@�G��~�R?���A)�C���@�G����
�#�
����C�\                                    BxqО4  "          @�
=@�����Q�?��A'�
C�ff@������ͽu�z�C��                                    BxqЬ�  
�          @��@�33�x��?uAC�{@�33��Q�\)��33C���                                    Bxqл�  �          @�z�@��R�hQ�?�G�AE�C�l�@��R�u�>aG�@
=C��                                    Bxq��&  
�          @��
@�p��[�?�\A���C�)@�p��q�?B�\@�C���                                    Bxq���  
Z          @��@����Mp�>�@���C��@����Mp���G����C�f                                    Bxq��r  �          @��H@��\�Fff?.{@�Q�C���@��\�J�H�B�\��z�C�aH                                    Bxq��  "          @�=q@����R?�A��C��R@���8��?��A-G�C��                                    Bxq��  
�          @��@����G�@*�HAٙ�C��
@����*�H@G�A��C�s3                                    Bxq�d  
�          @�(�@��H���R@B�\A��\C�B�@��H�0  @��A�z�C���                                    Bxq�"
  "          @�(�@��H���@A���C��
@��H�<(�?���A}�C�\)                                    Bxq�0�  

          @��H@���@��@
=A�
=C��3@���b�\?��HAg\)C��)                                    Bxq�?V  T          @��@�ff�.{@	��A��C��3@�ff�L��?��AS�C�ٚ                                    Bxq�M�  �          @�z�@���^{?�G�A=qC���@���g
=<�>�z�C��{                                    Bxq�\�  T          @��@�G��G�?�A_33C�z�@�G��XQ�?�@��
C�b�                                    Bxq�kH  
�          @�{@���E>�=q@(��C�O\@���C33����  C�y�                                    Bxq�y�  
Z          @�z�@�Q��Z=q>�
=@�(�C�1�@�Q��Y����\��  C�>�                                    Bxqш�  "          @��
@����l(�>\@qG�C�h�@����j=q��R��33C���                                    Bxqї:  "          @��@�G��~{�L�Ϳ�Q�C���@�G��qG���G��F�\C�^�                                    Bxqѥ�  T          @��H@g
=���\��p��Ep�C��f@g
=�u�ff��=qC���                                    BxqѴ�  	�          @�33@XQ���p���
=����C��)@XQ��s33�333��RC��{                                    Bxq��,  	�          @��H@}p����Ϳ}p���C��{@}p��o\)�z���ffC�XR                                    Bxq���  
�          @��
@z=q��33��Q��dz�C��3@z=q�dz���R��33C���                                    Bxq��x  
�          @��
@H����{��G����HC�l�@H���hQ��$z��ᙚC�h�                                    Bxq��  
�          @�(�@z=q�����5��
=C���@z=q�mp���G����C�=q                                    Bxq���  "          @��@�z��x��?�{A���C�e@�z���  ?B�\@���C�%                                    Bxq�j  
�          @�\)@�G��j�H?�
=A�
=C��R@�G�����?c�
A��C�P�                                    Bxq�  T          @�\)@�=q�`  ?ǮAs33C�E@�=q�r�\?z�@��C�(�                                    Bxq�)�  
�          @���@�p��L(�?�Q�A\��C���@�p��\��?
=q@��C�w
                                    Bxq�8\  �          @�@�33��=q?k�A�C��3@�33��p��B�\����C�:�                                    Bxq�G  T          @�@vff��?G�@���C�� @vff��\)��
=���HC�XR                                    Bxq�U�  
�          @���@~�R��\)>�(�@�=qC��3@~�R���E���(�C��R                                    Bxq�dN  �          @��@�33��33�u�z�C���@�33�����H�:=qC�4{                                    Bxq�r�  "          @��@�  ���?B�\@�z�C�L�@�  ���
�����S33C�R                                    Bxqҁ�  �          @�33@o\)��(�?fffA�\C�@ @o\)��
=��\)�0��C��)                                    BxqҐ@  
�          @���@g
=����?h��A33C�=q@g
=��(����R�B�\C���                                    BxqҞ�  "          @��@i����  �@  ��(�C��H@i�����Ϳ�z���=qC���                                    Bxqҭ�  
Z          @��@[���G���\)�1p�C���@[����H�G���  C���                                    BxqҼ2  "          @���@a���  �O\)��C��@a���(����H����C�E                                    Bxq���  �          @���@G���33�(�����C�b�@G����׿�{��p�C�O\                                    Bxq��~  �          @���@Z�H���ͽ�G���\)C�+�@Z�H��ff�����S�
C��                                     Bxq��$  T          @��\@G����\�u�Q�C�k�@G�����(����\C��                                     Bxq���  
�          @�=q@G
=���\�^�R�	�C�b�@G
=��{�ff��\)C���                                    Bxq�p  "          @�Q�@8Q���(��s33���C�B�@8Q����R�(����C�ff                                    Bxq�  
�          @�p�@g
=���?(��AC���@g
=�
==L��?5C�l�                                    Bxq�"�  �          @�{@�z�?��@5�A�R@��@�z���@8��A�Q�C�)                                    Bxq�1b  
�          @��R@�  @\)@C�
B  A�(�@�  ?�\)@b�\Bz�A��R                                    Bxq�@  
Z          @�\)@�
=?��@H��B�\A�\)@�
=?E�@]p�B=qA�                                    Bxq�N�  �          @�  @���?���@Q�B�A�=q@���>�(�@`��B�@�p�                                    Bxq�]T  �          @�  @��
?�\)@c�
B{Af�H@��
>.{@n�RB!��@�                                    Bxq�k�  �          @�p�@���?5@^{B\)A�H@��;W
=@a�B��C���                                    Bxq�z�  T          @�{@�녽��
@@  A�p�C��f@�녿Y��@7�A��C��q                                    BxqӉF  "          @�
=@�Q�?(�@(��A�
=@�ff@�Q콸Q�@-p�A�G�C�|)                                    Bxqӗ�  
�          @���@�?z�H?�(�A�=qA+33@�>�G�@
=qA�z�@��H                                    BxqӦ�  
Z          @�{@�G�?333@ ��A�=q@�@�G�<�@'
=A��H>�p�                                    Bxqӵ8  
�          @�ff@���?k�@{A�A"ff@���>��
@Q�A�{@e�                                    Bxq���  
�          @�
=@��>�\)@�HA�
=@C�
@����33@=qA�(�C��                                    Bxq�҄  
(          @���@��?��@��A���AR{@��?�@�RAՙ�@�(�                                    Bxq��*  �          @��@�=q��G�@G�A�=qC��@�=q�p��?���A���C���                                    Bxq���  T          @�\)@���!G�@(�A�G�C�� @����p�@��A�  C�9�                                    Bxq��v  �          @�\)@��=L��@!G�A��
?�\@�����@(�A�p�C���                                    Bxq�  r          @���@�{�s33@
�HA���C��{@�{��(�?�{A�{C�R                                    Bxq��  h          @�(�@�녾��@�RAǅC�~�@�녿�=q@�A�z�C�>�                                    Bxq�*h  T          @��
@��\�8Q�@z�A��HC�&f@��\���@�
A���C�5�                                    Bxq�9  "          @��@��ÿB�\?�33A�{C�q@��ÿ��H?��A�\)C���                                    Bxq�G�  T          @��@���E�?�  Ak\)C�%@����\)?�G�ADQ�C�e                                    Bxq�VZ  T          @���@�z῞�R?޸RA��C���@�z���?�\)AZ=qC���                                    Bxq�e   �          @���@�\)�:�H@z�A�C�f@�\)��ff@�
A�\)C��                                    Bxq�s�  
�          @��H@�zῌ��?�Q�A�G�C�AH@�z��ff?���A~ffC���                                    BxqԂL  	�          @���@��ÿ���@�
A��C�  @��ÿ�=q?��HA�(�C���                                    BxqԐ�  T          @�G�@��\����?\Aw33C��q@��\��(�?�\)A5��C�                                    Bxqԟ�  
�          @���@��\��p�?�\)A[�
C��)@��\�G�?c�
A��C���                                    BxqԮ>  
�          @���@����G�?��HAA��C���@��� ��?:�H@�\)C���                                    BxqԼ�  "          @�G�@�������?�  AH(�C���@������?8Q�@�C��q                                    Bxq�ˊ  T          @���@��Ϳٙ�?���A?\)C�Ff@��Ϳ���?:�H@��C�R                                    Bxq��0  "          @�=q@�33��z�?ǮAy��C�^�@�33�   ?��A-C��q                                    Bxq���  �          @��\@�z��G�?��AO�
C��@�z���\?Q�A{C���                                    Bxq��|  "          @��H@��
��  ?��HAh  C��q@��
��
?xQ�A�C��                                    Bxq�"  "          @��\@����˅?Q�AG�C��@����޸R>Ǯ@xQ�C�H�                                    Bxq��  T          @�=q@�
=����?�z�A`��C��@�
=��z�?��\A"�RC��\                                    Bxq�#n  �          @��@����p�?&ff@�\)C�@ @����>L��?�p�C��)                                    Bxq�2  
�          @��\@��׿�ff>��@��C���@��׿�{<#�
=�\)C���                                    Bxq�@�  T          @���@�ff���?L��@��RC��)@�ff���>�Q�@eC���                                    Bxq�O`  "          @�  @��\���>k�@  C�\@��\�Q쾣�
�J=qC�q                                    Bxq�^  
�          @�
=@�\)�  ?�\@�33C�\)@�\)��
��\)�333C��                                    Bxq�l�  T          @�\)@����Q�?   @���C��q@����(��L�;��HC���                                    Bxq�{R  
�          @�Q�@���  ?��\A$z�C�J=@����H?�R@�C�H�                                    BxqՉ�  
�          @���@�{����?�Q�Ag�
C���@�{�\?��A/\)C�,�                                    Bxq՘�            @�\)@����Q�?�33Ac
=C�Ф@����  ?��A+33C�B�                                    BxqէD  
�          @���@��ÿ�=q?��RAG�C�o\@�����?=p�@�ffC�=q                                    Bxqյ�  
�          @���@��\��{?��HABffC�e@��\�ff?8Q�@��
C�>�                                    Bxq�Đ  
�          @�Q�@����ff?��AT  C��@����=q?c�
AffC��{                                    Bxq��6  �          @��@��R���?��A|z�C�@��R�$z�?n{A\)C�O\                                    Bxq���  �          @�
=@�녿�G�?�z�A;�C���@�녿��R?0��@�C��)                                    Bxq���  T          @��@����z�?333@�  C�Z�@�����>#�
?�33C��R                                    Bxq��(  
�          @��@�\)��?�@�Q�C���@�\)��z�=u?
=C�aH                                    Bxq��  T          @���@�(���
=?+�@ָRC�q@�(���\>.{?�(�C��q                                    Bxq�t  "          @�33@����\)?�  A��C���@�����H>��@��C���                                    Bxq�+  �          @�(�@��H�ff?0��@׮C��@��H���=�Q�?W
=C��\                                    Bxq�9�  
�          @��H@���'�?333@�C��\@���-p�<�>�{C�'�                                    Bxq�Hf  
(          @��\@����1G�?n{A�C���@����:�H>aG�@��C���                                    Bxq�W  
Z          @�G�@���C�
?z�HA  C��f@���L��>B�\?�C�(�                                    Bxq�e�  S          @���@��
�QG�?�\)A3�C�S3@��
�\��>�\)@2�\C��q                                    Bxq�tX  T          @��@���J�H?z�@�
=C�q�@���Mp��aG����C�B�                                    Bxqւ�  
�          @���@���N{?�z�Ab�RC�c�@���^{?�@��RC�`                                     Bxq֑�  	�          @���@���6ff?�{A�{C�}q@���J=q?\(�A	��C�'�                                    Bxq֠J  s          @��@���X��?޸RA�{C�
@���mp�?Y��A�
C��                                    Bxq֮�  g          @��@����AG�?}p�AC��@����J=q>W
=@�C�O\                                    Bxqֽ�  
�          @�Q�@�  �Mp�>u@
=C��R@�  �J�H�\)��33C�#�                                    Bxq��<  
�          @��R@��R�r�\?�@��\C��q@��R�s33��
=��\)C��                                    Bxq���  "          @��R@~{�~{?uA33C��H@~{���H�#�
��ffC��                                    Bxq��  �          @��@�Q��W
=?��A0Q�C��f@�Q��aG�>u@�C���                                    Bxq��.  �          @�
=@����R�\?�Q�A@  C���@����^�R>�33@]p�C�4{                                    Bxq��  T          @�
=@��
�aG�?��A0(�C��@��
�k�>B�\?���C��                                    Bxq�z  
�          @��R@�(��J�H?��
A&=qC�˅@�(��U�>aG�@p�C�&f                                    Bxq�$   "          @��R@�\)�
=q?.{@��C���@�\)���=�?�G�C�L�                                    Bxq�2�  �          @��@�33�"�\?5@�33C��H@�33�(��=�\)?&ffC�S3                                    Bxq�Al  T          @�p�@����E?�33A��C�
@����]p�?�=qA1��C��f                                    Bxq�P  "          @�33@p���S33@{A�(�C�aH@p���s33?˅A�=qC�XR                                    Bxq�^�  
�          @���@��H�U?�
=A���C�b�@��H�mp�?��A-�C��                                    Bxq�m^  �          @�ff@\(���{?��HA�{C���@\(����?#�
@��C��H                                    Bxq�|  T          @�{@�G��R�\?��A���C�>�@�G��e?J=q@��C�f                                    Bxq׊�  �          @��@���B�\?�33A�Q�C��@���Z=q?���A0(�C�aH                                    BxqיP  
�          @���@�
=�HQ�?��A��C�� @�
=�`  ?��A(��C���                                    Bxqק�  
�          @�\)@�z��l��?��A`��C�{@�z��{�>�G�@�ffC�:�                                    Bxq׶�  
�          @�\)@r�\����?��A/�C�4{@r�\��G�=#�
>�(�C���                                    Bxq��B  
�          @�  @\)�z�H?�ffAP��C��)@\)���
>���@@  C��                                    Bxq���  
;          @��@|���z�H?�\)A\��C��)@|����z�>�p�@l(�C��R                                    Bxq��  "          @�  @\)�|��?�
=A>�\C��H@\)���
>8Q�?���C��                                    Bxq��4  T          @��@z�H��=q?�ffAO\)C��3@z�H����>�=q@*=qC�H�                                    Bxq���  �          @���@r�\��(�?�z�AaC�G�@r�\��33>�Q�@dz�C���                                    Bxq��  �          @���@�z��s33?�Ac
=C��R@�z�����>�ff@�
=C��                                     Bxq�&  T          @�=q@��\�c�
?\Ar=qC�G�@��\�tz�?��@��RC�G�                                    Bxq�+�  �          @�=q@��R�W�?�=qA|(�C�y�@��R�i��?5@��C�Z�                                    Bxq�:r  �          @��@{����?k�A��C�o\@{����H�\)����C�)                                    Bxq�I  
�          @�z�@�=q���?\(�AffC�33@�=q����8Q�޸RC��=                                    Bxq�W�  
(          @��@w���=q?#�
@ə�C��=@w���33��(����C��{                                    Bxq�fd  �          @�\)@�Q��c33?���AyG�C��H@�Q��u?0��@�(�C��                                    Bxq�u
  �          @�(�@����{�?�=qAQ��C�J=@�����(�>���@P��C��\                                    Bxq؃�  
�          @�@��H�l��?�ffAs�C���@��H�~{?(�@�(�C�Ǯ                                    BxqؒV  �          @�ff@���\)?�A:ffC�� @������>.{?�z�C�*=                                    Bxqؠ�  �          @�ff@�z��+�@A�Q�C���@�z��Fff?���AUC���                                    Bxqد�  �          @���@��R�E�?�(�A�(�C�k�@��R�Y��?k�A(�C�3                                    BxqؾH  �          @�{@���ff@�A���C���@���?�
=A�=qC��                                    Bxq���  T          @�
=@������@z�A��
C���@���
=?�\)A�(�C�                                    Bxq�۔  �          @�
=@����33@
=A�C��@���
=?�=qAv�\C��                                    Bxq��:  �          @��@�\)�Q�@ ��A��HC�e@�\)�+�?�z�A�C�ٚ                                    Bxq���  �          @��@��\��=q@��A�z�C��H@��\�33?�Q�A�{C�z�                                    Bxq��  �          @�Q�@�녿�=q?У�A{�C��@�녿�?���A7�C�t{                                    Bxq�,  �          @�G�@��׿��?�ffAn=qC���@������?�ffA�C�)                                    Bxq�$�  
�          @���@�z��p�?�\)A�Q�C��@�z���?�=qAL(�C�"�                                    Bxq�3x  T          @�=q@��Ϳ˅?��
Aj=qC�"�@��Ϳ�?���A'33C��H                                    Bxq�B  
�          @���@�p����?��A��\C�aH@�p��G�?��
AD(�C���                                    Bxq�P�  
�          @���@�ff��=q?��
A��
C���@�ff�p�?��AE�C���                                    Bxq�_j  "          @�G�@����-p�?�A���C�Ф@����Dz�?���A(��C�G�                                    Bxq�n  T          @���@�{�Vff?޸RA�(�C�5�@�{�k�?^�RA��C��3                                    Bxq�|�  �          @���@����N�R?У�A|Q�C��@����a�?L��@���C���                                    Bxqً\  "          @��R@�Q��[�?��
A���C�e@�Q��p��?fffA
�RC�)                                    Bxqٚ  "          @�{@�p��!G�@'�Aң�C��R@�p��E�?�
=A���C�N                                    Bxq٨�  
�          @��R@��R�'�@ ��A�C�xR@��R�I��?�ffA���C��                                    BxqٷN  "          @�ff@�  �U�@   A��HC��H@�  �n{?���A.�\C�<)                                    Bxq���  �          @�@���S33?�A��\C�
=@���i��?z�HA(�C���                                    Bxq�Ԛ  �          @�33@�Q��7
=?��
A�C���@�Q��Mp�?��
A"=qC��                                    Bxq��@  �          @�{@�33�,(�@�A��C�|)@�33�HQ�?�33AZ{C���                                    Bxq���  "          @��@���0  @�A���C�B�@���J�H?���AQ��C�o\                                    Bxq� �  T          @��R@�(��$z�@��A�(�C��@�(��B�\?ǮAs�C�
=                                    Bxq�2  �          @��@���(�@0  Aޏ\C��@���B�\@�A��HC�O\                                    Bxq��  T          @�\)@���p�@,��A׮C��@��%�@��A�Q�C�4{                                    Bxq�,~  "          @�Q�@�=q�p�@1G�AܸRC��3@�=q�4z�@	��A�z�C��)                                    Bxq�;$  
�          @�33@����2�\@ ��A�33C���@����Tz�?�  A�=qC��
                                    Bxq�I�  "          @��
@�p��HQ�@
=A�(�C�R@�p��g
=?��
Ah  C�(�                                    Bxq�Xp  T          @��H@��qG�?�=qAs
=C�Ǯ@���G�?(�@��HC�˅                                    Bxq�g  �          @��@�\)�o\)?�\)A�z�C�:�@�\)���H?fffA33C���                                    Bxq�u�  "          @�
=@�z��[�@#�
A�C�.@�z��|��?�33A��\C�,�                                    Bxqڄb  T          @��@��H�`  ?�Q�A���C���@��H�w�?��
A ��C�#�                                    Bxqړ  �          @�{@�p��U?�Q�A`��C�33@�p��e?z�@�33C�7
                                    Bxqڡ�  T          @�{@����S33@ ��A��
C��\@����l(�?��A1G�C�c�                                    BxqڰT  s          @�{@��\�\��@�A���C���@��\�vff?�A7\)C�+�                                    Bxqھ�  
�          @�z�@a��~�R@(�A�G�C���@a����R?�33A\��C�1�                                    Bxq�͠  "          @�p�@q��u�@��A�C�K�@q���=q?�Q�Aap�C���                                    Bxq��F  
�          @�33@�z��@  @=qA�33C�Ф@�z��`  ?�{A�C��
                                    Bxq���  "          @��H@����S33@#�
AиRC�n@����tz�?�
=A��C�U�                                    Bxq���  �          @��@c�
�]p�@FffA��
C���@c�
���@
=qA�G�C�q�                                    Bxq�8  
�          @��@k��8Q�@Tz�B�C��3@k��g
=@!G�A�ffC��                                    Bxq��  �          @��@h����H@mp�B!ffC�0�@h���P��@@��A�33C��                                    Bxq�%�  T          @�=q@J�H�#33@���B6=qC�� @J�H�_\)@Y��B=qC�(�                                    Bxq�4*  
�          @�G�@B�\�$z�@�ffB:  C��=@B�\�aG�@\(�B��C�n                                    Bxq�B�  �          @�=q@k����@l��B ��C���@k��N{@@  A�z�C�ff                                    Bxq�Qv  4          @��@��*=q@H��B�C���@��Vff@��A��RC��f                                    Bxq�`  �          @�ff@�  �\��@�A�ffC�t{@�  �z�H?�z�A[�
C���                                    Bxq�n�  �          @�{@�\)�u�?��
Ap  C��\@�\)���H?
=q@���C�                                      Bxq�}h  �          @�z�@������?��\AF�\C�.@�������>aG�@Q�C���                                    Bxqی  �          @��
@��R�g�?��A�{C��H@��R�~{?p��A�C�O\                                    Bxqۚ�  �          @��H@�33�i��?�
=A��HC�'�@�33����?xQ�A=qC��                                    Bxq۩Z  �          @��H@�\)�U�@��A��
C��@�\)�q�?�\)AY�C�                                      Bxq۸             @���@�z��j=q?�p�A�\)C�AH@�z��~{?G�@�  C�{                                    Bxq�Ʀ  �          @���@����g�?�Q�A���C��q@����\)?}p�A=qC���                                    Bxq��L  �          @�G�@c�
�qG�@!�A��C���@c�
����?\At��C��f                                    Bxq���  �          @���@S�
�~�R@"�\A�
=C��@S�
���?�p�An=qC�:�                                    Bxq��  �          @��H@qG��w
=@\)A�(�C�'�@qG�����?�(�A@��C���                                    Bxq�>  T          @�p�@r�\�G�@P  B33C�G�@r�\�tz�@Q�A�G�C�`                                     Bxq��  
�          @��
@I�����@{A�ffC�S3@I����
=?���AU�C��                                    Bxq��  �          @���@H����\)@
=qA���C��H@H�����
?}p�A\)C�b�                                    Bxq�-0  �          @�p�@AG�����@33A�\)C��H@AG���ff?�{A-C��{                                    Bxq�;�  T          @�{@O\)���@�A���C�%@O\)���H?�\)A-�C��                                    Bxq�J|  �          @��
@G
=���@2�\A���C���@G
=��(�?ٙ�A���C���                                    Bxq�Y"  �          @��\@8����=q@&ffA���C��H@8�����\?���Ag�C�p�                                    Bxq�g�  �          @���@7���ff@>�RA�RC�.@7�����?�A�  C�n                                    Bxq�vn  �          @��@e��w
=@1�Aޏ\C�l�@e���?޸RA���C�z�                                    Bxq܅  �          @�=q@W���33@ffA�
=C��@W����?�G�AH��C�9�                                    Bxqܓ�  �          @��@[����\?޸RA��C�+�@[���(�?(�@�=qC�AH                                    Bxqܢ`  �          @���@E�����?�\)A��RC�R@E���33?0��@�=qC�0�                                    Bxqܱ  
�          @���@B�\��p�?��Aw�C���@B�\����>�{@U�C���                                    Bxqܿ�  �          @���@:�H��=q?�ffAPQ�C��{@:�H���=u?#�
C�#�                                    Bxq��R  �          @�  @+����R?�z�A�p�C��{@+���G�?.{@ٙ�C�                                    Bxq���  �          @��@8����\)?�p�A�z�C��
@8������?   @���C��q                                    Bxq��  �          @��@333��z�?�ffAvffC��@333���
>�z�@3�
C�W
                                    Bxq��D  �          @�@333��ff?�(�Ao�
C�ff@333��p�>�  @ ��C��
                                    Bxq��  �          @���@XQ�����?�Q�Al��C�+�@XQ���  >���@W
=C�w
                                    Bxq��  �          @���@X�����?��AS
=C�
@X����  >.{?�(�C���                                    Bxq�&6  �          @��@c�
��G�?�=qAV=qC��
@c�
���>W
=@Q�C�8R                                    Bxq�4�  T          @�G�@]p�����?�=qA,��C��
@]p���zὣ�
�G�C�T{                                    Bxq�C�  �          @�  @dz���G�?��AP��C�޸@dz���\)>8Q�?�C�E                                    Bxq�R(  �          @�\)@n�R��ff?�G�A#33C�˅@n�R��=q��\)�:�HC�e                                    Bxq�`�  T          @��@����s�
?�Q�Ah��C�J=@�������>�G�@�(�C�k�                                    Bxq�ot  T          @�p�@����X��?\Ayp�C��@����j=q?�R@�=qC��
                                    Bxq�~  �          @���@��׽L��?���A���C��R@��׾�ff?��
A�33C��                                    Bxq݌�  �          @���@�ff�333?�(�A�
=C�&f@�ff��{?��RAy�C��
                                    Bxqݛf  t          @��@�p���\)?ٙ�A��HC��=@�p���G�?�\)Ad(�C��                                     Bxqݪ  �          @�=q@�p����R?���A��C�=q@�p���{?��RAM��C�`                                     Bxqݸ�  �          @��H@�(��z�?�\A��C��f@�(����?�Q�AD��C���                                    Bxq��X  �          @���@��H�?�p�A���C�T{@��H�p�?��A>=qC���                                    Bxq���  �          @�=q@�33���?�{A�{C�:�@�33�2�\?���AH��C�aH                                    Bxq��  �          @��
@��(�?�33A��RC�AH@��5?�p�AJ=qC�c�                                    Bxq��J  �          @�{@�Q��7
=?�A�Q�C��3@�Q��P  ?��A8��C�q                                    Bxq��  �          @�
=@�Q��<(�?�
=A�p�C�s3@�Q��U�?�\)A4��C��f                                    Bxq��  �          @���@�Q��=p�@�
A��HC�` @�Q��XQ�?��RAFffC��3                                    Bxq�<  
�          @�Q�@�
=�G�?�{A��RC��f@�
=�_\)?�G�A!G�C���                                    Bxq�-�  �          @���@�(��dz�@
=A��C��R@�(��\)?��A3
=C��                                    Bxq�<�  �          @�ff@�G��k�@�
A�
=C��)@�G���z�?�ffAJ{C�'�                                    Bxq�K.  �          @��R@����fff?��A�  C�\)@����|(�?Q�@�z�C�R                                    Bxq�Y�  �          @�  @~{�p��@{A\C�C�@~{����?�
=A]�C�u�                                    Bxq�hz  �          @��@w��q�@$z�A�
=C�ٚ@w����?\Ak33C���                                    Bxq�w   �          @��@�G��o\)@�A�ffC��q@�G���
=?��AN�\C��                                    Bxqޅ�  �          @�
=@�=q�mp�@�A��C��3@�=q��p�?�G�AB�HC�(�                                    Bxqޔl  T          @�G�@��\�l(�@�RA�(�C���@��\��ff?���A_
=C�                                    Bxqޣ  �          @�G�@��e@p�A�{C���@���33?��HA_33C���                                    Bxqޱ�  �          @\@���U�@!�A��C�|)@���w
=?���As�
C�b�                                    Bxq��^  �          @��@�\)�]p�@�A�{C�*=@�\)�y��?�(�A8��C�z�                                    Bxq��  �          @���@��e�@�A��\C��@��\)?��A%�C��R                                    Bxq�ݪ  �          @�G�@�  �j=q@33A�z�C��@�  ��(�?��
AC�C���                                    Bxq��P  T          @���@�
=�qG�@	��A���C��@�
=��ff?�{A)�C��3                                    Bxq���  �          @�  @�
=�mp�@
=A��RC�Q�@�
=��(�?�=qA&ffC���                                    Bxq�	�  �          @���@�z��i��@ffA�z�C�N@�z����?��A)��C�                                    Bxq�B  �          @�
=@��
�   >aG�@p�C�H@��
�{��(����HC�!H                                    Bxq�&�  �          @��H@�=q�ff>�p�@i��C�R@�=q�
=��  �(�C�                                    Bxq�5�  �          @�(�@�33�4z�?Q�A Q�C���@�33�<(�=u?�RC��                                    Bxq�D4  �          @��H@���C�
?=p�@�=qC��@���I����\)�:�HC���                                    Bxq�R�  �          @�(�@�  �@  ?8Q�@���C���@�  �E����
�O\)C�/\                                    Bxq�a�  �          @�{@����HQ�>�(�@�p�C��@����H�þ����tz�C��                                    Bxq�p&  �          @��R@�
=�O\)>\@j�HC�s3@�
=�N�R�����=qC��                                     Bxq�~�  �          @�z�@�33�Mp�=�\)?&ffC�Ff@�33�G
=�E���C���                                    Bxqߍr  �          @���@z�H�e�@(�A��C���@z�H����?�
=A=C��                                    Bxqߜ  �          @�G�@w
=�r�\@33A��
C��H@w
=��{?}p�Ap�C�O\                                    Bxqߪ�  �          @���@tz�����?�p�A���C���@tz����?\(�Ap�C�y�                                    Bxq߹d  �          @���@k��z=q@0��A�G�C���@k���  ?��A{�
C���                                    Bxq��
  �          @��H@p����G�@#33A�ffC�y�@p����=q?�z�AU��C��)                                    Bxq�ְ  �          @�33@s�
���@=qA�Q�C�ff@s�
��33?�  A=�C��\                                    Bxq��V  �          @�=q@p  ���H@��A�(�C�C�@p  ���H?��AC�
C���                                    Bxq���  �          @��@�  �p  @$z�Aȣ�C�o\@�  ��G�?��RAd(�C��                                     Bxq��  �          @��@��\�e�@-p�A�=qC�` @��\���?�A�{C�4{                                    Bxq�H  �          @���@��H�c33@  A��RC�]q@��H��Q�?�p�A<z�C��q                                    Bxq��  �          @Å@���Tz�@p�A�C�  @���r�\?�  A=��C�P�                                    Bxq�.�  �          @�z�@���\��@�A�(�C��3@���x��?��A+
=C���                                    Bxq�=:  �          @\@�  �3�
?��A�(�C�O\@�  �Mp�?�=qA%�C��H                                    Bxq�K�  �          @�=q@�\)�@  ?�p�A�Q�C�xR@�\)�Vff?^�RA(�C�
=                                    Bxq�Z�  �          @�33@�ff�-p�@z�A���C���@�ff�N�R?\Af�\C�o\                                    Bxq�i,  �          @Å@�p��   @ ��AÅC���@�p��E�?�  A���C��q                                    Bxq�w�  �          @\@u�n{@8Q�A�\C��{@u��33?��A�
=C��
                                    Bxq��x  �          @��H@j=q�p  @FffA���C�'�@j=q��{?��RA���C��H                                    Bxq��  �          @\@�Q��Q�@G
=A�(�C�Z�@�Q��\)@Q�A���C��\                                    Bxq��  �          @\@����@  @;�A���C�޸@����j�H@�A��
C�{                                    Bxq�j  �          @���@����Dz�@Tz�B{C�Q�@����u@Q�A��C�#�                                    Bxq��  �          @��H@}p��G
=@Z=qBffC��=@}p��z=q@��A�=qC���                                    Bxq�϶  �          @�33@�33�Fff@R�\Bz�C�o\@�33�w�@A�=qC�Q�                                    Bxq��\  �          @�33@���7�@J=qA��
C�L�@���g�@�A�G�C�'�                                    Bxq��  �          @�33@tz��S�
@W�B�RC��@tz����H@
=A�z�C��                                    Bxq���  �          @�33@a��e@[�B	{C�@ @a���(�@A�z�C�k�                                    Bxq�
N  �          @Å@XQ����\@8��A��C��{@XQ����R?�Q�A��
C��R                                    Bxq��  �          @�33@O\)�s33@]p�B
�RC�7
@O\)��33@33A�(�C��R                                    Bxq�'�  T          @��H@=p���
=@8��A�Q�C��q@=p����H?���Ao
=C�
                                    Bxq�6@  �          @\@k��P  @e�B=qC�H�@k���33@#�
AǙ�C���                                    Bxq�D�  �          @�=q@u��r�\@(��A��HC���@u����
?�G�AiC��)                                    Bxq�S�  �          @�=q@i����ff?���A6=qC��@i����33����33C�w
                                    Bxq�b2  �          @��@g
=��Q�?^�RAQ�C��
@g
=��=q����33C�l�                                    Bxq�p�  �          @�=q@W���ff@33A�
=C���@W����?�  A��C�9�                                    Bxq�~  �          @��H@Vff���@C33A�z�C��q@Vff���?���A���C��q                                    Bxq�$  �          @��
@W
=��G�@HQ�A�Q�C�޸@W
=��  ?�z�A��C���                                    Bxq��  �          @��
@j=q�xQ�@@  A���C��f@j=q����?�=qA��HC�b�                                    Bxq�p  �          @\@qG��x��@0  A�p�C��@qG����?�=qAq��C��                                    Bxq�  �          @\@w��s�
@.�RA��
C���@w����?�=qAq�C��                                     Bxq�ȼ  �          @��@dz��~�R@*=qA��
C��@dz����?�(�Ac�C��                                    Bxq��b  �          @���@u��q�@,(�A���C��=@u���(�?�ffAn�HC��{                                    Bxq��  �          @�  @s�
�u�@&ffA�C�j=@s�
����?���A_\)C�p�                                    Bxq���  �          @��
@u�l(�@�\A�  C�R@u��?�
=A<Q�C�P�                                    Bxq�T  �          @�\)@��\�g
=>��@�\)C�
@��\�fff�����=qC�"�                                    Bxq��  �          @��@�G��qG�?�ffA&{C�^�@�G��z=q�#�
��Q�C��
                                    Bxq� �  �          @�Q�@�Q��^�R?�{A�z�C�]q@�Q��r�\?z�@��\C�(�                                    Bxq�/F  �          @�(�@��\�6ff@1G�A���C�Y�@��\�`  ?�\)A�ffC��f                                    Bxq�=�  �          @�  @����z�H?�G�A
=C�+�@�����G�����C���                                    Bxq�L�  �          @�\)@�=q�z�H?��AJ=qC���@�=q��(�>#�
?�=qC�&f                                    Bxq�[8  �          @�Q�@�����Q�?���A��C��f@������?!G�@\C�t{                                    Bxq�i�  �          @�\)@�(��e@=qA�Q�C���@�(����
?��AK
=C���                                    Bxq�x�  �          @�ff@p���r�\@(��A�Q�C�ff@p����(�?�(�Ae�C�XR                                    Bxq�*  �          @�p�@g
=���\@�\A�ffC��@g
=���?��A$��C�4{                                    Bxq��  �          @���@p�����\@ ��A��C�U�@p����\)?G�@��
C��                                    Bxq�v  �          @�(�@}p��p  @��A��C�O\@}p����R?��A%��C��H                                    Bxq�  �          @�p�@��H�n{@
=A�G�C��R@��H���?z�HAz�C�<)                                    Bxq���  �          @�@����k�?��
A��C���@�������?+�@θRC�Z�                                    Bxq��h  �          @�{@���e�?�p�Af�\C��{@���vff>���@z=qC��                                    Bxq��  �          @��@���s33?��
A Q�C���@���|(���Q�\(�C��                                    Bxq���  �          @�(�@�G��e?Tz�A�C�Ф@�G��k��k��p�C�z�                                    Bxq��Z  �          @�(�@����_\)?Tz�A�C��\@����e��B�\��33C�33                                    Bxq�   �          @�{@�p��fff@z�A���C���@�p���G�?uA�HC��{                                    Bxq��  �          @���@����XQ�@\)A��\C��)@����w�?�
=A9p�C���                                    Bxq�(L  �          @�z�@�\)�O\)@�A�z�C�AH@�\)�s33?�z�A^�RC�                                    Bxq�6�  �          @�33@�z��\(�@  A�G�C�%@�z��{�?�A9p�C�8R                                    Bxq�E�  �          @��@�=q�Y��@(�A��C��@�=q�}p�?�\)AX��C��                                     Bxq�T>  �          @�(�@s33�"�\@o\)B{C�)@s33�`��@6ffA��C��f                                    Bxq�b�  �          @�@j�H���@|��B'ffC��@j�H�^�R@E�A���C�>�                                    Bxq�q�  �          @�
=@z=q��@���B)�RC�J=@z=q�Fff@QG�B{C�˅                                    Bxq�0  �          @���@�z����@mp�B�C���@�z��W�@7
=A�C�l�                                    Bxq��  �          @\@��\�3�
@N�RB   C��f@��\�g�@�A��C��                                    Bxq�|  �          @�(�@���>{@G�A��HC��)@���p  @Q�A���C���                                    Bxq�"  �          @��H@��R�I��@Dz�A�(�C�� @��R�y��@�A���C��)                                    Bxq��  �          @Å@����_\)@2�\A�  C���@�������?�z�A|��C���                                    Bxq��n  �          @���@����~{?���A��
C���@������
?5@��HC�n                                    Bxq��  �          @�G�@mp����@�A��
C��R@mp����?@  @�RC�C�                                    Bxq��  �          @�\)@\�����?�(�Ae�C�j=@\�����H=u?(�C��H                                    Bxq��`  T          @���@q���  @�A�=qC��
@q�����?�33A0��C��                                    Bxq�  �          @���@��R�<��@(Q�A��
C�@ @��R�e?�z�A��C���                                    Bxq��  �          @��H@�ff�O\)@<��A�ffC�1�@�ff�}p�?�\)A���C�Z�                                    Bxq�!R  �          @\@�Q��W�@.�RA�p�C��{@�Q�����?�\)Aw�C�S3                                    Bxq�/�  �          @\@�
=�L��@%A��C�/\@�
=�tz�?��Aj=qC���                                    Bxq�>�  �          @�=q@�=q�U�@*=qA�  C�/\@�=q�~{?ǮAn�RC��R                                    Bxq�MD  �          @�=q@����P��@&ffA�
=C���@����xQ�?\Ah��C�Q�                                    Bxq�[�  �          @Å@����Z=q@�RA�\)C��
@����z=q?�\)A)C���                                    Bxq�j�  �          @��
@����N�R@!�A��
C�C�@����u?�(�A]C��                                    Bxq�y6  �          @��
@�
=�c�
@
=qA��C���@�
=��G�?�G�A��C��R                                    Bxq��  �          @�33@��h��@ffA���C�J=@���33?k�A\)C���                                    Bxq䖂  �          @��@���^{@#33A�33C���@����=q?�33AQ�C���                                    Bxq�(  �          @�@�(��j=q?�A�ffC���@�(���G�?#�
@�p�C�t{                                    Bxq��  �          @���@����c�
?�G�A>ffC��3@����q�>\)?�ffC��                                    Bxq��t  !          @�@�=q�[�?�p�A�{C�H�@�=q�r�\?�R@�G�C��                                    Bxq��  �          @�ff@����Fff?�A�{C�*=@����a�?fffA�C�q�                                    Bxq���  �          @Å@�ff�Y��?�\)A��HC�\@�ff�s�
?B�\@�ffC���                                    Bxq��f  �          @�33@�\)�A�?��A�Q�C�XR@�\)�[�?J=q@�\)C��)                                    Bxq��  �          @\@�
=�9��@\)A�G�C�4{@�
=�aG�?�  Ad(�C���                                    Bxq��  �          @�33@������@333A���C�c�@����@�A�Q�C���                                    Bxq�X  �          @�@����=q@{A�
=C�33@��� ��?���A��
C��\                                    Bxq�(�  �          @�{@�Q쿫�@:=qA��C��
@�Q��
=q@
=A�ffC��R                                    Bxq�7�  �          @�ff@������@{A��C���@����1�?ٙ�A
=C��                                    Bxq�FJ  �          @�z�@����
=@2�\A�33C��H@����6ff@G�A�Q�C�1�                                    Bxq�T�  �          @�z�@��
�#33@{A���C��q@��
�G
=?��AIG�C�^�                                    Bxq�c�  �          @���@��.{@7
=A���C��@��]p�?�33A�ffC�                                    Bxq�r<  �          @�(�@��R�>�R@5�Aޣ�C�"�@��R�l��?��A��C�(�                                    Bxq��  �          @Å@����5�@<(�A��C���@����fff?�Q�A�Q�C�                                    Bxq又  �          @\@�=q�@  @(��A�=qC�]q@�=q�j�H?���As�C��                                    Bxq�.  �          @�p�@����H@I��A��C�"�@����\?�G�AbffC���                                    Bxq��  �          @ƸR@���Q�@UB(�C���@����?��HA���C�
=                                    Bxq�z  �          @�
=?�\)��@`��B	�C���?�\)����?���A�=qC��                                    Bxq��   �          @�
=@  ��=q@n�RB��C��)@  ��Q�@p�A���C���                                    Bxq���  �          @�\)@,����@[�B�
C���@,������?��A��C���                                    Bxq��l  �          @ȣ�@%��=q@\(�B��C��f@%���?�{A��C��=                                    Bxq��  �          @ȣ�@'���
=@`��B��C�(�@'����H?���A�p�C��                                    Bxq��  �          @�G�@)�����@fffB��C�|)@)�����@33A��\C�>�                                    Bxq�^  �          @�G�@7
=����@fffB
=C��@7
=��@A�C�w
                                    Bxq�"  �          @ə�@Z=q��33@Tz�A�33C���@Z=q��?��A�
=C�]q                                    Bxq�0�  �          @Ǯ@g��s33@Tz�B33C��@g�����?�(�A��C���                                    Bxq�?P  �          @ȣ�@���333@s�
BC��q@���vff@/\)AυC�n                                    Bxq�M�  �          @��@}p��@  @uB��C�g�@}p����@,��A��
C�'�                                    Bxq�\�  �          @�G�@Z=q����@X��B��C�@Z=q���?���A�ffC�c�                                    Bxq�kB  �          @ə�@�{�9��@l(�B�\C��=@�{�z=q@%A�ffC�|)                                    Bxq�y�  �          @�=q@��\���@a�B��C��@��\�\(�@$z�A��C���                                    Bxq戎  �          @��H@mp��Z=q@p��BC���@mp���p�@\)A�ffC�                                      Bxq�4  �          @��
@J=q�y��@z=qB��C��f@J=q��@�RA�p�C�S3                                    Bxq��  �          @�=q@h���^�R@q�Bp�C�%@h����  @�RA�p�C�~�                                    Bxq洀  �          @˅@n{�XQ�@w�BG�C��q@n{��{@%A�
=C���                                    Bxq��&  �          @�(�@n�R�g
=@k�BffC���@n�R���H@A���C���                                    Bxq���  �          @���@p  �X��@z=qB��C���@p  ���R@'�A��C��                                    Bxq��r  �          @�z�@���5�@���B��C���@���~{@:=qA�33C���                                    Bxq��  �          @��
@�G��7�@~�RB  C�P�@�G��\)@6ffA�33C��                                    Bxq���  �          @���@�  �)��@|��B  C�f@�  �q�@8��Aי�C�0�                                    Bxq�d  �          @��@�{���@h��B
C�xR@�{�\(�@+�A�{C�޸                                    Bxq�
  �          @�@�{��@[�B  C���@�{�P��@!G�A�Q�C�H�                                    Bxq�)�  �          @θR@�  �#�
@c33BffC�ٚ@�  �c�
@"�\A�
=C��                                    Bxq�8V  �          @�{@�G��6ff@dz�B�\C��3@�G��vff@p�A�\)C��q                                    Bxq�F�  �          @�
=@�{�8��@l��B�RC�y�@�{�{�@#�
A�33C�>�                                    Bxq�U�  �          @�{@���(�@W�A�z�C��\@���Y��@��A���C���                                    Bxq�dH  �          @�p�@�G��
=@b�\BG�C��@�G��XQ�@%�A�Q�C�e                                    Bxq�r�  �          @�{@���!G�@mp�B
=C���@���e@,(�A��C��                                    Bxq灔  �          @�@���/\)@eB  C���@���p��@   A��C�B�                                    Bxq�:  �          @���@�  �{@`  B��C�C�@�  �^�R@   A�{C��f                                    Bxq��  �          @�{@����#33@r�\BffC�T{@����i��@0  A�Q�C��
                                    Bxq筆  �          @�ff@��\�   @��BQ�C�  @��\�l(�@AG�A߅C��                                    Bxq�,  �          @Ϯ@�33���@y��B��C�9�@�33�c33@8��A�C�0�                                    Bxq���  �          @�  @��R���@[�A�=qC�
@��R�9��@'�A��
C�p�                                    Bxq��x  �          @У�@��H��@HQ�A�C���@��H�<��@�\A�ffC��\                                    Bxq��  �          @��@�G��
=@O\)A�p�C�%@�G��C�
@�A�p�C���                                    Bxq���  �          @��H@�G��G�@W
=A�p�C��\@�G��@��@   A���C�(�                                    Bxq�j  �          @��
@��
��\@eB{C�!H@��
�G�@-p�A�=qC�Y�                                    Bxq�  �          @ҏ\@��H����@`  BG�C�h�@��H�!�@4z�A�(�C�T{                                    Bxq�"�  �          @ҏ\@��ÿ���@J�HA�\C��\@����#�
@p�A���C���                                    Bxq�1\  �          @ҏ\@�z���
@7
=A�(�C�=q@�z��(Q�@ffA��C��                                     Bxq�@  �          @Ӆ@�ff���
@5A�G�C�S3@�ff�(Q�@�A���C���                                    Bxq�N�  �          @�(�@�  �У�@7�A͙�C��@�  �   @
=qA�z�C�B�                                    Bxq�]N  �          @�33@�G���{@,��A�G�C�9�@�G���H@   A��C���                                    Bxq�k�  �          @љ�@�(���33@]p�A��C��3@�(���R@2�\AɮC���                                    Bxq�z�  �          @љ�@��
����@W�A��
C��3@��
�(��@(Q�A��
C��=                                    Bxq�@  �          @�z�@�\)��z�@u�B{C�w
@�\)�'�@HQ�A�\)C��
                                    Bxq��  �          @��
@�z��@8��A�ffC�3@�z��+�@
=A��C�C�                                    Bxq覌  �          @�33@�p���\)@�RA�ffC�7
@�p��!G�?�(�AM�C�xR                                    Bxq�2  �          @���@��H�   ?�33AG�C�e@��H�5�>��@�C��                                    Bxq���  �          @�=q@���1�?�ffA\)C�N@���>{=L��>���C���                                    Bxq��~  �          @љ�@����8��?}p�A�C��)@����C33�L�;�
=C��                                    Bxq��$  �          @�G�@�=q�XQ�?�@��\C�\)@�=q�W
=�(�����C�t{                                    Bxq���  �          @���@��
�j�H>��@e�C���@��
�e�^�R��p�C��                                    Bxq��p  �          @���@���S�
?���A�C�q�@���_\)���Ϳc�
C�Ǯ                                    Bxq�  T          @�  @���j�H�L�;��C�Ф@���[���ff�8��C���                                    Bxq��  �          @�p�@�����=q��{� Q�C�ٚ@����\(��(�����C�,�                                    Bxq�*b  �          @��H@fff���
�����C�� @fff�X���]p���C�aH                                    Bxq�9  �          @�G�@"�\��\)�0���مC�"�@"�\�_\)��
=�1Q�C�
=                                    Bxq�G�  �          @���@2�\���H�
=��p�C��q@2�\�s�
�i�����C��                                    Bxq�VT  �          @�G�@33��\)�����\)C�h�@33�x���vff�!�C�O\                                    Bxq�d�  �          @�G�?�(������;���=qC�4{?�(��^{�����?=qC���                                    Bxq�s�  �          @\@�����ͿTz����C��@���u�33��  C�xR                                    Bxq�F  �          @���@��H�����ͿfffC�]q@��H���
�У��z�RC�ff                                    Bxq��  �          @�=q@�p����
=�?�
=C�� @�p��z=q�����Ip�C�B�                                    Bxq韒  �          @���@����\)>k�@(�C�E@����tzῗ
=�3�
C��                                    Bxq�8  �          @���@���z�H�#�
��G�C���@���j�H�����Tz�C��
                                    Bxq��  �          @�@dz���(��&ff��33C��
@dz���33��R��(�C���                                    Bxq�˄  �          @���@>�R���ÿ0����C�S3@>�R��
=�=q����C��R                                    Bxq��*  �          @�33@L�������8Q���z�C��)@L������
=���
C���                                    Bxq���  �          @���@N{��Q��R���C�Y�@N{��
=�ff���RC��q                                    Bxq��v  �          @Å@7����\�z����C�(�@7�������H���
C�}q                                    Bxq�  �          @�
=@C�
��(�������C�c�@C�
��Q����(�C�Z�                                    Bxq��  �          @�p�@Mp�����    <�C��3@Mp�����{��ffC��\                                    Bxq�#h  �          @�@W
=��p�>�z�@+�C�~�@W
=��{�Ǯ�j�RC��                                    Bxq�2  �          @��H@Z=q����>���@5�C�{@Z=q������  �e�C��\                                    Bxq�@�  �          @���@Z=q��\)?E�@�=qC��@Z=q��ff�u�{C���                                    Bxq�OZ  �          @�{@~{��33?�  A9C�e@~{���������
C���                                    Bxq�^   �          @Å@u���{?޸RA��\C�b�@u�����=#�
>��C�`                                     Bxq�l�  �          @���@q��}p�@(�A�Q�C�ٚ@q���=q?^�RA��C��\                                    Bxq�{L  �          @Å@�������?��@�{C��@�����{���� z�C��                                    Bxq��  �          @�@{���{?@  @�
=C��q@{����Ϳ}p��33C�                                      Bxqꘘ  �          @�
=@�
=��33?��AB�RC�\@�
=���þ�{�L(�C�|)                                    Bxq�>  �          @�p�@��\����?�  A��C���@��\���
=�Q�?W
=C��
                                    Bxq��  �          @�p�@�����?�
=A|��C�,�@����{=�Q�?Q�C��                                    Bxq�Ċ  �          @�p�@��H�|(�@��A��\C�
=@��H����?L��@�C�                                    Bxq��0  �          @�z�@hQ���G�@�RA��HC��@hQ�����?B�\@�(�C�AH                                    Bxq���  �          @�@z=q���
@�A�p�C�@z=q��
=?E�@�p�C���                                    Bxq��|  �          @Å@�33��33?�33A~�RC���@�33��p�=L��>�C�o\                                    Bxq��"  �          @�(�@�����
=?�
=A1��C��R@��������G���33C�AH                                    Bxq��  �          @\@����p�?�
=AZffC�y�@�����;B�\���C���                                    Bxq�n  �          @�ff@�\)�z=q?�Q�AaG�C��q@�\)��p����
�Q�C��{                                    Bxq�+  �          @���@��xQ�?�(�A:�RC�h�@�������z��0  C�Ǯ                                    Bxq�9�  �          @�=q@�{��(�?�AX��C��R@�{����L�Ϳ���C��\                                    Bxq�H`  �          @�
=@����u�?��A�33C��R@�����Q�>�{@K�C�Y�                                    Bxq�W  �          @�p�@�(����?���A4(�C�@ @�(���녾���uC���                                    Bxq�e�  �          @�ff@�������?   @���C�}q@�����������&ffC��q                                    Bxq�tR  �          @�33@QG���G�?�33A`��C�޸@QG�����\�p  C�L�                                    Bxq��  �          @�z�@C33��p�?˅A~�HC���@C33����  ���C��                                    Bxq둞  �          @���@33��  @'�A�\)C���@33��(�?+�@�=qC��\                                    Bxq�D  �          @�@0  ����@>�RA�{C�P�@0  ��33?�A6ffC�/\                                    Bxq��  �          @�{@{��G�@3�
A�ffC�Ff@{����?k�A(�C��3                                    Bxq뽐  �          @�@6ff���@�RA��C��R@6ff���>���@Mp�C�w
                                    Bxq��6  �          @�{@0  ���H?�G�AD��C�+�@0  ��{�:�H��\C��                                    Bxq���  �          @�z�@XQ���{?Y��A=qC�� @XQ���p��u�  C��                                    Bxq��  �          @�@\)���>��@|(�C�h�@\)���Ϳ��R�BffC���                                    Bxq��(  �          @�ff@|(��G
=@HQ�A���C���@|(����?�p�A�=qC��                                    Bxq��  �          @���@i���A�@]p�B\)C�#�@i�����@z�A��RC��                                    Bxq�t  �          @�@4z��Z�H@r�\B"
=C���@4z���33@{A�
=C�                                    Bxq�$  �          @���@\)��\)@�  Bs��C�3@\)�mp�@���B/{C��\                                    Bxq�2�  �          @�G�@G��G
=@�33B+��C��@G���@'�AΏ\C��
                                    Bxq�Af  �          @��H@�
=�h��@{A��C���@�
=���R?.{@θRC���                                    Bxq�P  �          @�G�@xQ���  ?�=qA���C�0�@xQ���(�=��
?@  C��q                                    Bxq�^�  �          @���@r�\��{@ffA���C��@r�\��p�>�33@S�
C��                                    Bxq�mX  �          @���@}p���G�?�\)AS\)C�\)@}p������Q��_\)C��
                                    Bxq�{�  �          @�Q�@�����(�?Tz�@�p�C�N@�������fff�
�\C�Z�                                    Bxq스  �          @�  @z�H����>\)?�ffC�o\@z�H��������x��C�b�                                    Bxq�J  �          @��R@1��~{@7
=A���C�}q@1���Q�?���A8��C��                                    Bxq��  �          @���@333��ff@S33B�\C���@333��z�?�
=A\(�C�Ff                                    Bxq춖  �          @�G�@K���=q?#�
@��
C��@K�����{�P(�C�j=                                    Bxq��<  �          @��@S�
��G�?�(�Ab�HC�S3@S�
��������
C���                                    Bxq���  �          @��
@X����p�?��RAh  C�H@X����z��
=����C�e                                    Bxq��  �          @���@������׽��
�J=qC��f@����z=q��p����HC�+�                                    Bxq��.  �          @��@j=q���R�aG���C���@j=q��Q������RC�Q�                                    Bxq���  �          @���@fff�����L�Ϳ��C�n@fff��33��\��
=C��                                    Bxq�z  �          @���@���z�H>�(�@�ffC�+�@���q녿�\)�.�HC��\                                    Bxq�   �          @��@u��ff?c�
A	C�j=@u��{�h�����C�l�                                    Bxq�+�  �          @��@E���G��L�;�ffC���@E���(�� �����RC�ٚ                                    Bxq�:l  �          @�ff@hQ�����?�Q�A��C��
@hQ����\>.{?�z�C�4{                                    Bxq�I  �          @�@�=q��>���@O\)C�!H@�=q�~�R��=q�R=qC��
                                    Bxq�W�  �          @�{@���p  >k�@\)C�H�@���b�\��  �D(�C��                                    Bxq�f^  �          @���@�Q���  ��\)�0  C�ff@�Q��dz��=q���C��                                    Bxq�u  T          @�@�\)�5�@��A��C���@�\)�\(�?\(�AG�C�H                                    Bxq탪  �          @��@��\�Vff@�A��C��@��\��G�?k�A��C�z�                                    Bxq�P  �          @�
=@�33�[�@)��A���C�\@�33��{?��A'�
C�.                                    Bxq���  �          @�Q�@�=q�e�@Q�A�Q�C�'�@�=q��(�?
=q@�ffC�                                      Bxq���  �          @��R@�(��`��?�z�A�  C���@�(��~�R>�Q�@aG�C���                                    Bxq��B  �          @�@�Q��j=q?�ffA$  C�y�@�Q��q녾���ffC��                                    Bxq���  �          @�@����o\)>�(�@�{C�H�@����g
=����)C��                                    Bxq�ێ  �          @�z�@�ff�tz�?   @��
C���@�ff�mp�����%�C�R                                    Bxq��4  �          @�z�@�(��Y��?B�\@��HC��)@�(��Z�H�&ff��{C���                                    Bxq���  �          @���@����\)?���Av{C�` @����:�H>�@���C���                                    Bxq��  �          @���@����
=?�p�A=G�C�33@����)��>B�\?�C���                                    Bxq�&  �          @�p�@����!G�?�
=A`(�C��
@����8Q�>���@<��C�b�                                    Bxq�$�  �          @�\)@���ff?�p�A��RC�9�@���'�?=p�@�p�C��f                                    Bxq�3r  T          @�\)@�G��
�H?�  A�  C���@�G��,(�?:�H@���C���                                    Bxq�B  �          @�Q�@��׿�z�@�A��C�f@����&ff?�A4Q�C��)                                    Bxq�P�  �          @��@�\)�G�@�A�33C�s3@�\)�1G�?��
AB�HC��                                    Bxq�_d  �          @Å@���
�H?��
A���C��@���,��?@  @��HC��q                                    Bxq�n
  �          @�z�@�
=��\)@33A�{C���@�
=�!�?�\)A(Q�C��3                                    Bxq�|�  �          @���@�  ��@z�A�
=C��@�  �\)?�z�A-��C�Ф                                    Bxq�V  �          @�p�@����33@�A��C���@����?�AT��C��                                    Bxq��  �          @�{@��\��\)@+�AͮC���@��\�$z�?��A��HC�                                      Bxq  �          @�ff@��\��@33A�  C�\@��\� ��?У�Au��C��                                    Bxq�H  �          @��
@�(��>�R?�33AT��C��@�(��R�\=�\)?!G�C��f                                    Bxq���  �          @��H@��N{?��
AiG�C�j=@��dz�=�Q�?L��C�                                    Bxq�Ԕ  �          @\@�(��S33?��RAb�RC���@�(��g�<#�
=���C���                                    Bxq��:  �          @��H@��
�+�?�{A�z�C�*=@��
�Mp�?��@���C���                                    Bxq���  �          @�=q@�
=�:�H?�{A��HC���@�
=�Z�H?   @�
=C��R                                    Bxq� �  �          @�33@����   ?�=qAqG�C�aH@����;�>�
=@z�HC���                                    Bxq�,  �          @��
@�33�p�?�z�A{�C���@�33�:�H?   @��RC��\                                    Bxq��  �          @�33@�\)�&ff?�p�A��C��@�\)�E�?   @�
=C���                                    Bxq�,x  �          @�(�@�p��#�
@G�A�(�C��3@�p��J=q?G�@�33C�C�                                    Bxq�;  �          @Å@��\�
=q?�z�A/33C�j=@��\���>#�
?��
C�*=                                    Bxq�I�  �          @�(�@�����
?&ff@�33C��@�������Q��W�C��)                                    Bxq�Xj  �          @�(�@�  �
=>�33@S33C��=@�  �z�����C��                                    Bxq�g  �          @�(�@�녿�
=<#�
>�C��@�녿�G��J=q����C��\                                    Bxq�u�  �          @�33@��H��
=��(���=qC��{@��H���Ϳ���%G�C�z�                                    Bxq�\  �          @�33@�녿�
=�8Q���G�C��{@�녿�(������MG�C��                                    Bxq�  �          @�z�@�{������Q��3�C��@�{���
��33��Q�C��=                                    Bxq  �          @�33@��Ϳ��������C���@��Ϳz�H��=q�#�
C�Ff                                    Bxq�N  �          @��@�ff��=q�u�p�C�q@�ff��G��У��z=qC��                                    Bxq��  �          @��H@�Q����Y��� ��C�W
@�Q쿢�\���
�hQ�C���                                    Bxq�͚  �          @��H@�Q���H�}p��p�C���@�Q쿑녿�{�v{C�g�                                    Bxq��@  �          @���@��ÿ˅�z����C�H�@��ÿ�����Q��6�HC�)                                    Bxq���  �          @�=q@��
��?z�@���C�5�@��
���
��\)�!G�C��\                                    Bxq���  
�          @���@�(���\)?0��@��
C��R@�(����>#�
?�  C��
                                    Bxq�2  T          @�Q�@�z��33>�p�@`  C�H@�z�� �׿����HC�+�                                    Bxq��  T          @�{@��H���H>L��?�
=C�Y�@��H��{�&ff��Q�C��                                    Bxq�%~  T          @�ff@�p��У�?\)@��RC�  @�p����H�8Q��(�C���                                    Bxq�4$  �          @��@��׿��R����ffC��@��׿�{��G��EG�C��                                     Bxq�B�  �          @��R@�  ��(�>�
=@��\C���@�  ��G��u��C��H                                    Bxq�Qp  T          @��R@��׿���=�?�C���@��׿�{��\��z�C�XR                                    Bxq�`  �          @�ff@�\)���>�  @��C�|)@�\)��  ��(����C��=                                    Bxq�n�  �          @�ff@�G����>B�\?�C�� @�G���������u�C��R                                    Bxq�}b  �          @��R@��R��=q?�G�A=qC��H@��R��z�>�ff@��C�                                    Bxq��  �          @�  @������\?�z�A��C��=@����޸R?��AH��C�J=                                    Bxq�  �          @��@��R�Q�?���Aw\)C���@��R��?��A((�C��                                    Bxq�T  �          @�G�@�����?��AT��C���@�����\?�ffA z�C�                                    Bxq��  T          @Å@��\����?��
A��C�` @��\����?�(�A^�RC���                                    Bxq�Ơ  �          @���@�p���Q�?���Am��C�C�@�p���  ?��\A>ffC�33                                    Bxq��F  �          @�{@��R���?�{Aqp�C��@��R���?��
A>�RC��{                                    Bxq���  T          @�z�@��\�333?޸RA�(�C���@��\����?��\A?�C�Y�                                    Bxq��  �          @ƸR@�p��G�?�
=A{�
C�AH@�p���
=?�
=A/�C�7
                                    Bxq�8            @�@�(��c�
?�\)AtQ�C���@�(���G�?�=qA!�C�Ф                                    Bxq��  �          @�\)@�{�Y��?�z�Aw33C���@�{��p�?���A&�RC���                                    Bxq��  �          @Ǯ@�Q�k�?�\)AK33C���@�Q쿷
=?Tz�@�{C�S3                                    Bxq�-*  �          @ȣ�@Å�Tz�?���A!��C�!H@Å��p�?!G�@�Q�C�T{                                    Bxq�;�  �          @�=q@��
����?��A&=qC�f@��
���H?\)@�=qC�G�                                    Bxq�Jv  �          @��@�=q��33?��
A;
=C���@�=q�˅?&ff@��C��f                                    Bxq�Y  �          @��H@�녿��R?���AIG�C�=q@�녿��H?333@��
C��                                    Bxq�g�  �          @�=q@�  ���H?�=qAB�\C�%@�  ��33?\)@�=qC�9�                                    Bxq�vh  �          @��@�녿���?�33A(��C���@�녿ٙ�>�@�p�C�.                                    Bxq�  �          @�G�@�녿��?��A\)C��@�녿�33>��@l��C�aH                                    Bxq�  �          @�=q@��R��{?��A@  C�n@��R��>��@���C��q                                    Bxq�Z  �          @��
@�\)��(�?�{AD��C���@�\)���>�@��C�!H                                    Bxq�   �          @��
@�Q��=q?��AJ�\C���@�Q���?\)@�G�C���                                    Bxq�  �          @˅@�Q쿰��?\A\��C���@�Q��z�?@  @��C�33                                    Bxq��L  �          @�33@��ÿ���?�33ApQ�C��3@��ÿ޸R?z�HA�C��{                                    Bxq���  �          @�33@�
=��\)?�{A�{C��3@�
=����?�A*=qC��H                                    Bxq��  �          @ʏ\@��R��33?���Ak
=C�]q@��R��(�?Q�@�C�ٚ                                    Bxq��>  �          @��@�{�B�\?�(�A��C�Y�@�{��ff?�
=AQG�C���                                    Bxq��  �          @ə�@���@  @�\A���C�aH@����=q?��RA[33C���                                    Bxq��  �          @ə�@����Q�?�A�p�C�S3@����{?��A�C�>�                                    Bxq�&0  �          @�  @�\)���
?�  A]��C�"�@�\)��=q?aG�A{C��{                                    Bxq�4�  �          @Ǯ@�{��\)?�  A_
=C��@�{��?W
=@���C�!H                                    Bxq�C|  �          @���@�Q쿦ff?��\A;
=C��H@�Q��p�?\)@��HC��3                                    Bxq�R"  �          @�  @�\)���\?��\A<  C�H@�\)�ٙ�?�@��C��                                    Bxq�`�  �          @ƸR@�\)���?��HA4Q�C���@�\)�\?
=@�  C�޸                                    Bxq�on  �          @ȣ�@�=q�h��?��\A:�\C��
@�=q����?8Q�@��
C���                                    Bxq�~  T          @�  @�=q�h��?���A&{C�� @�=q����?��@�  C��                                    Bxq�  �          @�  @�G��
=q?�G�A_�C�w
@�G���z�?�{A#\)C��{                                    Bxq�`  �          @�\)@����@  ?��
A=C�u�@�����  ?O\)@�C�.                                    Bxq�  �          @�(�@��R�333?�  A<Q�C���@��R��Q�?L��@�Q�C�^�                                    Bxq�  �          @��H@��Ϳp��?�  A�
C�w
@��Ϳ��>��@�p�C��{                                    Bxq��R  �          @��H@�  ��
=>�p�@_\)C��3@�  ��녿
=q��(�C��                                    Bxq���  �          @�33@�Q��%=�?�
=C�j=@�Q������� Q�C�aH                                    Bxq��  �          @�=q@����{���
�E�C���@����
�H��Q��5��C�H�                                    Bxq��D  �          @��@��R��>W
=?��HC�0�@��R��z�=p��߮C�                                    Bxq��  �          @��@��ÿ��
=�\)?#�
C�s3@��ÿ�\)�@  ���HC�1�                                    Bxq��  �          @���@��H��Q�=p�����C�9�@��H�:�H�����7�
C�q�                                    Bxq�6  �          @���@�G��zῇ��"=qC��H@�G���{�����(�C�R                                    Bxq�-�  T          @�(�@��
��p�>#�
?��HC��=@��
��{�&ff���HC�S3                                    Bxq�<�  �          @��@�  ��  >�@��C�!H@�  ���þ#�
��(�C�˅                                    Bxq�K(  �          @�(�@�  ��
=>�p�@Z�HC�q�@�  ��(��W
=�   C�Ff                                    Bxq�Y�  �          @Å@����ff?\)@�
=C��@����Q�<#�
>�C�e                                    Bxq�ht  �          @�(�@��ÿfff?��@��C��f@��ÿ�ff=��
?=p�C��                                    Bxq�w  �          @Å@���z�H?E�@�ffC�]q@����(�>k�@�C�>�                                    Bxq��  �          @�p�@�녿Q�?L��@�  C�!H@�녿���>�{@HQ�C�ٚ                                    Bxq�f  �          @�p�@�=q�z�H>�@�G�C�k�@�=q��=q�#�
��p�C��3                                    Bxq�  �          @���@�=q�&ff?5@���C��@�=q�h��>���@EC��                                     Bxq�  �          @�(�@����z�?G�@陚C�@ @����aG�>�
=@|��C��H                                    Bxq��X  �          @��@��׿k�?aG�A
=C���@��׿�(�>�33@QG�C�AH                                    Bxq���  �          @�z�@�
=����?J=q@��C�S3@�
=��
=>��?��C�E                                    Bxq�ݤ  �          @��@��ÿaG�?Q�@���C��{@��ÿ�z�>��
@<��C���                                    Bxq��J  �          @���@�녿J=q?!G�@��C�Ff@�녿�  >B�\?�ffC�U�                                    Bxq���  T          @�p�@�G��n{?W
=@�33C���@�G����H>��
@<(�C�S3                                    Bxq�	�  �          @�p�@��׿aG�?}p�A(�C���@��׿��R>��@�33C�*=                                    Bxq�<  �          @�ff@\�Y��?Tz�@�\)C��@\���>�{@J=qC��\                                    Bxq�&�  �          @�@�녿k�?G�@�\)C��{@�녿�>��@�C���                                    Bxq�5�  �          @�@�ff���?�p�A7�
C��@�ff���?z�@�z�C��H                                    Bxq�D.  �          @Ǯ@�{�:�H?˅An�HC�|)@�{���?���A
=C�g�                                    Bxq�R�  �          @�
=@�33���@z�A��
C���@�33����?��HA�C�7
                                    Bxq�az  �          @�p�@��
�L��?�z�Az=qC�"�@��
��p�?��A#�
C���                                    Bxq�p   �          @��@��
�J=q?�z�Az�\C�'�@��
��p�?���A$(�C��                                    Bxq�~�  �          @�@�=q�L��?��A�(�C�{@�=q�˅?��A@��C�\)                                    Bxq�l  �          @Ǯ@��H����?�z�A���C���@��H����?�Q�A/\)C�9�                                    Bxq��  �          @�
=@�ff��33@p�A��RC�P�@�ff�33?�z�ARffC��                                    Bxq���  �          @Ǯ@�G��333@	��A�{C���@�G��У�?���Aj=qC�&f                                    Bxq��^  �          @�@�=q�#�
@FffA�p�C��3@�=q��(�@.�RA�=qC�L�                                    Bxq��  �          @�{@��\>8Q�@vffBQ�@@��\��z�@^�RB	(�C��H                                    Bxq�֪  �          @�@��
��@�=qB2�\C��@��
��@a�B�C�޸                                    Bxq��P  �          @�p�@n{��Q�@���BPQ�C�>�@n{�%�@��B'z�C��q                                    Bxq���  �          @�(�@����ff@�
A��
C�W
@���\)?�AW
=C��                                    Bxq��  �          @��
@�\)���
@{A���C�4{@�\)���?�p�A:ffC�.                                    Bxq�B  �          @�(�@����
=@%�AǅC��=@����R?���AqC���                                    Bxq��  �          @��H@������@
=A�
=C��3@���?�AXQ�C�S3                                    Bxq�.�  �          @��H@����{?�(�A9p�C��@����!�<��
>8Q�C���                                    Bxq�=4  �          @���@�
=��p�@
=A�\)C��H@�
=�7
=?��A,��C���                                    Bxq�K�  �          @�=q@�=q�ٙ�@��A�  C�%@�=q� ��?�=qA%C�Y�                                    Bxq�Z�  �          @���@�=q����@33A�p�C���@�=q�,(�?aG�A{C���                                    Bxq�i&  �          @��H@�=q�#33?�p�AaC�0�@�=q�<��=�Q�?O\)C�~�                                    Bxq�w�  �          @���@���G�?@  @�\)C�@���ff��(���Q�C��{                                    Bxq��r  �          @��@��H���Ϳ�
=�5p�C��R@��H�������\)C��H                                    Bxq��  �          @���@����p������_33C��@��Ϳ��R����Q�C���                                    Bxq���  �          @�(�@����{���R�<��C�  @����G����H�]�C�H                                    Bxq��d  �          @��@����(��>Ǯ@j�HC�<)@����   �k��
=qC�Ф                                    Bxq��
  �          @�(�@�G��8Q�?��A$z�C��
@�G��C33��(��~�RC��                                    Bxq�ϰ  �          @�=q@���  ?��A�Q�C�7
@���8Q�?�@�G�C�h�                                    Bxq��V  �          @��H@�33��@'
=A�p�C�#�@�33�B�\?��AG
=C��)                                    Bxq���  �          @�G�@��Ϳ��\@'�A�(�C�{@����Q�?�
=A�=qC��H                                    Bxq���  �          @�@��\���?�
=A���C�Q�@��\��z�?�AU��C�5�                                    Bxq�
H  w          @��@����(Q�?O\)@�z�C���@����,�Ϳ\)��(�C�AH                                    Bxq��  
�          @�p�@�=q���\)��ffC�=q@�=q�����G��Q�C��{                                    Bxq�'�  �          @�(�@�ff�.{�#�
��Q�C�@�ff�������Hz�C�>�                                    Bxq�6:  �          @\@�=q�J=q?\)@��C�
=@�=q�B�\�����C���                                    Bxq�D�  �          @��
@�Q��Y��?+�@���C��@�Q��S33�����!p�C�N                                    Bxq�S�  �          @���@��H�u�?���AG\)C��@��H�\)�333��33C��H                                    Bxq�b,  �          @ƸR@����C�
?��A�=qC�K�@����e>#�
?�p�C�1�                                    Bxq�p�  �          @�@�ff�5@�A�G�C��@�ff�g�?&ff@�G�C��\                                    Bxq�x  �          @ʏ\@�{�C�
@\)A���C��@�{�x��?0��@�\)C��f                                    Bxq��  �          @�ff@���;�@(�A�
=C���@���hQ�>��H@��\C���                                    Bxq���  �          @�{@�(��mp�@   A�33C��=@�(����R�u���C���                                    Bxq��j  T          @�z�@�z����\?�@�\)C��@�z��tz���
�]C��                                    Bxq��  �          @�33@���~�R?!G�@���C�~�@���q녿�33�K�C�8R                                    Bxq�ȶ  �          @���@����=p�?��HA-�C���@����J�H�Ǯ�b�\C��                                    Bxq��\  �          @��@��
�~�R��Q�L��C�ff@��
�Z�H�G���{C�u�                                    Bxq��  �          @�33@��H��ff?�A*ffC���@��H��p����
�:�RC���                                    Bxq���  �          @��H@��\����?E�@޸RC�B�@��\��G����up�C��)                                    Bxq�N  �          @��
@c�
���R?#�
@��\C�+�@c�
����   ����C�                                      Bxq��  �          @�33@Mp���ff?
=q@��
C�>�@Mp���Q��p�����C�P�                                    Bxq� �  �          @���@�33��?O\)@�{C���@�33��
=�˅�k�C�=q                                    Bxq�/@  �          @�\)@�(�����?��\A33C��
@�(���\)��(��5G�C�                                    Bxq�=�  �          @�=q@�p�����?�{AH  C��)@�p���z�k���\C���                                    Bxq�L�  �          @ə�@|������>Ǯ@g�C�˅@|����33� �����C�!H                                    Bxq�[2  �          @���@@  ���������z�C�@@  ���H�E���C�U�                                    Bxq�i�  �          @���@/\)����>���@J�HC�@/\)����z���p�C��                                    Bxq�x~  
�          @�
=@�\)����?�G�A�z�C��@�\)��\)�����HC��                                    Bxq��$  
�          @���@�G��P  @�A���C�4{@�G��}p�>�p�@aG�C�z�                                    Bxq���  �          @��@�z��q�@�A�Q�C���@�z����R>B�\?޸RC�J=                                    Bxq��p  �          @���@�����ff@\)A��C���@�����Q�\)��G�C��3                                    Bxq��  �          @�{@q����\?�  A~�HC��@q���녿aG���p�C�^�                                    Bxq���  �          @�ff@{�����?�G�A~{C��{@{����ÿY����=qC��                                    Bxq��b  �          @�p�@z=q��33?�(�AS�C�n@z=q��{����$z�C�/\                                    Bxq��  �          @��@n{��(�?޸RA}G�C��=@n{��33�k����C��                                    Bxq���  �          @У�@l�����\?ٙ�Ar�RC��@l����Q쿆ff��C��{                                    Bxq��T  �          @�  @g���p�?�G�AW�C�}q@g���\)���\�4Q�C�S3                                    Bxq�
�  �          @θR@@  ����?�AK
=C�AH@@  ��  ���
�[�
C�S3                                    Bxq��  �          @�
=@:=q���H?�33AG�C�� @:=q��G��˅�c\)C���                                    Bxq�(F  �          @�ff@��  ?�  A�HC�� @��{�Q���
=C��                                    Bxq�6�  
�          @�@AG���(�?�33Ap  C��H@AG������G��6ffC�e                                    Bxq�E�  �          @�{@H�����H@	��A�Q�C���@H����\)�0���ƸRC��f                                    Bxq�T8  �          @θR@AG���=q@33A�  C���@AG���(��fff��\)C�q                                    Bxq�b�  �          @���@z�H����@�\A�p�C���@z�H��p��&ff��{C��R                                    Bxq�q�  �          @�Q�@h�����?���A*=qC�e@h����(��˅�c�C���                                    Bxq��*  �          @�  @^�R���\?�33A$z�C���@^�R����
=�p��C���                                    Bxq���  �          @�\)@\�����?E�@�33C�@ @\����=q��
��ffC��                                    Bxq��v  �          @θR@xQ����R?���A"ffC��@xQ���33�\�\(�C�U�                                    Bxq��  �          @���@<����=q?:�H@��C��R@<���������RC���                                    Bxq���  �          @�{@tz�����?��AQ�C���@tz������{�g�C���                                    Bxq��h  �          @θR@�(����?Y��@��HC�Y�@�(���{��(��S�
C���                                    Bxq��  �          @�{@�����?333@ȣ�C���@���|�Ϳ��R�V�HC�t{                                    Bxq��  �          @���@�Q��|(�>��
@:�HC���@�Q��dz���H�y�C�C�                                    Bxq��Z  �          @��
@����s�
?ٙ�Ax��C���@������;�����C���                                    Bxq�   �          @�ff@a���녿J=q��33C���@a����W���G�C�
                                    Bxq��  �          @�=q@[����\���\�<  C�@[��n�R�j�H���C�J=                                    Bxq�!L  �          @�G�@�ff�:�H?�G�A  C��@�ff�A녿(����C�xR                                    Bxq�/�  �          @�Q�@���N{?޸RA��C���@���j=q����(�C��                                    Bxq�>�  �          @ə�@�{�[�?8Q�@�33C�K�@�{�Tz῏\)�$��C��R                                    Bxq�M>  �          @�Q�@�\)�Mp�?�\)A%�C�9�@�\)�U�(���ÅC��)                                    Bxq�[�  �          @�p�@���(�?˅Ap��C��
@���:=q>�?��HC���                                    Bxq�j�  �          @�  @��H�8Q�=��
?:�HC��3@��H�!G���33�Up�C�^�                                    Bxq�y0  �          @�Q�@���O\)�aG��z�C�#�@���+�����
=C�z�                                    Bxq���  �          @�{@���p  �B�\��ffC���@���HQ������RC�B�                                    Bxq��|  �          @�p�@�z��S�
?#�
@��RC���@�z��K�����+
=C�&f                                    Bxq��"  �          @�@�z��`  ?�{AL  C�.@�z��mp���R��Q�C�g�                                    Bxq���  �          @��@�p��H��?�A.�RC�\)@�p��R�\�����33C��q                                    Bxq��n  �          @�ff@��R�O\)?�G�A�Q�C�s3@��R�k��\)���C��R                                    Bxq��  �          @�p�@��`��?���A�  C���@��}p��W
=���HC��                                    Bxq�ߺ  �          @ƸR@�\)�Mp�@0��A�
=C�8R@�\)��p�?:�H@ڏ\C���                                    Bxq��`  �          @�  @vff����?�p�A�  C��R@vff��G������33C�b�                                    Bxq��  �          @���@Q����R@�A�\)C�k�@Q���G������B�\C��                                    Bxq��  �          @�G�@_\)�xQ�@P��A��C��{@_\)��  ?O\)@���C�n                                    Bxq�R  �          @�z�@���Y��@�RA�
=C�˅@����{>\@_\)C�޸                                    Bxq�(�  �          @�(�@�{��
=?+�@�G�C�c�@�{����(��|��C�\)                                    Bxq�7�  �          @�=q@z�H���?�\)AK�C�@z�H��ff���H�3�C��                                    Bxq�FD  �          @��H@[���?O\)@�p�C��@[������(����\C��                                     Bxq�T�  �          @�33@9����p�?���AeC��@9�����R��Q��R{C��)                                    Bxq�c�  �          @��
@333��?�A�  C��H@333���H��  �4z�C�N                                    Bxq�r6  �          @�(�@#33��(�?�  AZ�HC�.@#33��33��{�j�HC�<)                                    Bxq���  T          @���@3�
��G�?��A_�C�s3@3�
��G����
�]C�p�                                    Bxq���  �          @��
@QG���  ?\A]��C���@QG����ÿ�33�K�C��                                     Bxq��(  �          @ʏ\@z����\?�z�A�{C�L�@z���Q쿠  �5C���                                    Bxq���  �          @�\)���
��p�?�{A��C�y����
���ÿ��H�YC��                                     Bxq��t  �          @�  �&ff����?���A�G�C��
�&ff���������L��C���                                    Bxq��  �          @���>W
=���@
=qA��C�
=>W
=��(���
=�.ffC���                                    Bxq���  �          @��>��
��p�@ffA��
C���>��
���Ϳ��\�:ffC��H                                    Bxq��f  �          @�=q?�p���G�@�A��
C��)?�p���  ��G��9p�C��f                                    Bxq��  T          @�Q�.{��p�?��
A��HC���.{��\)��ff�f�RC���                                    Bxq��  �          @�  >�ff����?��AH(�C�#�>�ff���H��p����C�4{                                    Bxq�X  �          @��?z���Q�?B�\@�z�C��f?z������{��C��                                    Bxq�!�  �          @�?˅��p�?+�@���C�:�?˅���
�(Q���C��f                                    Bxq�0�  T          @ʏ\?���\?0��@ə�C�  ?������$z���  C��                                    Bxq�?J  �          @�{@�{�[�@p�A��C���@�{���\=���?fffC�9�                                    Bxq�M�  �          @θR@�ff���@ ��A�{C�s3@�ff�;�?z�@���C�h�                                    Bxq�\�  �          @�\)@�����(�@��A�  C�@����(��?��A  C��H                                    Bxq�k<  �          @�
=@��R�.{@z�A�  C�@��R�Y��>�33@HQ�C��                                    Bxq�y�  �          @�ff@���X��?�A���C�Q�@���u�aG��   C���                                    Bxq���  �          @θR@�{���@A�C��{@�{�:�H?+�@�
=C�p�                                    Bxq��.  T          @Ϯ@���n{?޸RAy�C���@�����\����G�C�]q                                    Bxq���  �          @�\)@�=q���?���A!��C��
@�=q��zῙ���+�
C���                                    Bxq��z  T          @�
=@�p����\?�ffAG�C�  @�p���ff��  �W
=C�g�                                    Bxq��   T          @�  @�
=���?G�@�(�C�z�@�
=����������RC�k�                                    Bxq���  �          @�\)@��
��?z�HA33C���@��
��  ��{�fffC��                                    Bxq��l  T          @���@I�����R?�ffA:�HC��)@I�����H��  �{�
C�Ff                                    Bxq��  �          @�G�@xQ����H?�ffA�G�C�Y�@xQ���=q��  ��HC��R                                    Bxq���  �          @ҏ\@U��{?��AD��C��H@U���
��z��l(�C��                                    Bxq�^  �          @��H@0�����?��A=qC���@0����G������C�<)                                    Bxq�  �          @�=q@r�\��G�@�
A�G�C�.@r�\����@  �ָRC�&f                                    Bxq�)�  �          @Ӆ@I����@   A�33C��f@I����
=�
=���C�j=                                    Bxq�8P  �          @ҏ\@�Q����R?���A�C�ٚ@�Q����׿B�\�ٙ�C��                                    Bxq�F�  �          @�=q@���|��?��A{C��@���z�H���%C�"�                                    Bxq�U�  �          @��H@�=q�W�?Q�@�RC�ff@�=q�S33�����=qC��=                                    Bxq�dB  �          @�33@��
�`��?�Q�AJffC�j=@��
�p  ��R���C��\                                    Bxq�r�  �          @�z�@�
=�b�\?���A��C��=@�
=�e�k�� ��C�Z�                                    Bxq���  �          @���@�33�Y��?��HA��C��@�33�z�H����ffC��q                                    Bxq��4  
�          @���@��R�N�R@?\)A�z�C�z�@��R��=q?Y��@�=qC�|)                                    Bxq���  �          @�ff@����E@HQ�A߮C�<)@�������?��A
=C��3                                    Bxq���  
~          @�@��H���@`��B {C�l�@��H�xQ�?�p�Aq�C�S3                                    Bxq��&  0          @ָR@��\�{�@�A���C��@��\����aG���Q�C���                                    Bxq���  T          @�@x����=q?���A
=C�{@x�����\��33��Q�C��{                                    Bxq��r            @�p�@�\)���H?G�@ָRC���@�\)��\)�G����C��H                                    Bxq��  0          @�p�@������?��\A=qC��{@�����\��ff�}��C�7
                                    Bxq���  F          @�p�@��
����?�Q�AG�
C��3@��
���ÿ�
=�FffC���                                    Bxq�d  0          @�@�
=����?��
AU�C�]q@�
=��
=����2�HC�*=                                    Bxq�
  T          @ҏ\@���~�R@�RA�  C�q�@����{    �#�
C�)                                    Bxq�"�  
�          @��@�
=�S�
@,��A£�C�'�@�
=���?
=q@�\)C���                                    Bxq�1V  �          @�=q@�{�c�
@��A�\)C�q@�{��=q>L��?�(�C�n                                    Bxq�?�  "          @��@��H���?ٙ�Ap��C�<)@��H��33�\(���C�w
                                    Bxq�N�  �          @�=q@�{��33@G�A��C�n@�{��������  C��{                                    Bxq�]H  �          @��
@��H�~�R@A��C�H�@��H��\)�����9��C��R                                    Bxq�k�  �          @��H@�ff��33@
=A�(�C�Ф@�ff��녿�\���RC�aH                                    Bxq�z�  �          @��H@�ff��\)@�
A��C���@�ff���þ���fffC��{                                    Bxq��:  �          @�33@c33��{?��A�33C�.@c33���Ϳ�z��"�\C���                                    Bxq���  �          @Ӆ@l(����\?�\)A�
=C��@l(����������33C�o\                                    Bxq���  �          @ҏ\@x������@�A��C��R@x����ff�0������C�g�                                    Bxq��,  �          @���@���ff@7�A�G�C���@��=q?�\)A��C�~�                                    Bxq���  �          @�
=@��H�0��@$z�A�
=C���@��H����?���A�\)C��f                                    Bxq��x  T          @���@�p���@S33A�\)C��{@�p��^{?��HAn=qC���                                    Bxq��  "          @�\)@�Q���
@Z=qA��
C��@�Q��Tz�?�Q�A�ffC�z�                                    Bxq���  T          @�{@���\@?\)AծC�Z�@��E?���A]C��)                                    Bxq��j  �          @�p�@����
=@HQ�A�33C���@���E?�G�AuC��3                                    Bxq�  �          @�p�@�G��@r�\B  C��@�G��q�@ffA�33C��{                                    Bxq��  �          @��@�p��Q�@\��A��
C���@�p��vff?�33Af=qC��f                                    Bxq�*\  �          @ҏ\@��
�*=q@q�B�C�
=@��
��Q�?��A~{C���                                    Bxq�9  T          @��H@����W
=@2�\A��HC��f@������\?�@��RC�Ff                                    Bxq�G�  �          @�=q@�\)�|��@,(�A�Q�C�H�@�\)����>8Q�?��
C���                                    Bxq�VN  "          @Ӆ@�33�b�\@/\)A�(�C���@�33��ff>�(�@p��C��R                                    Bxq�d�  �          @Ӆ@����p��@>{A�Q�C�&f@�����Q�?�\@�{C��                                    Bxq�s�  �          @��H@�(���Q�@J�HA�\C��=@�(����\?
=q@�
=C���                                    Bxq��@  �          @ҏ\@��\�p��@L��A�33C�|)@��\��z�?333@�33C��                                     Bxq���  "          @љ�@i����{@A�Aޣ�C��R@i�����
>L��?޸RC��                                    Bxq���  �          @��@W
=���
@N{A뙚C�H@W
=���
>��@�C�j=                                    Bxq��2  �          @љ�@��H��\)@7
=A�G�C�f@��H��33>#�
?��C�]q                                    Bxq���  �          @���@y����=q@?\)A�\)C�@y����  >aG�?�33C�K�                                    Bxq��~  T          @�Q�@P�����
@_\)B�\C�Y�@P����=q?�R@�
=C�/\                                    Bxq��$  T          @ҏ\@;�����@c33B\)C�R@;����\?�\@�  C�W
                                    Bxq���  �          @��H@7����@Y��A��\C�N@7���p�>�\)@=qC��                                    Bxq��p  �          @�33@���{@S33A�{C�H�@�����<#�
=�Q�C���                                    Bxq�  �          @ҏ\@p����
@7
=AθRC�T{@p���녾�ff�}p�C�                                    Bxq��  "          @�G�@n{���H@FffA��C�N@n{���\>���@(��C�xR                                    Bxq�#b  T          @�=q@����u�@1�AȸRC��f@�����
=>���@'
=C��=                                    Bxq�2  "          @�=q@P����@0  A�C���@P����(����R�0��C�f                                    Bxq�@�  �          @љ�@Dz�����@UA�33C��{@Dz���
=>���@8Q�C�q                                    Bxq�OT  
�          @��
@����~�R@Y��A��C���@�����?=p�@�C��                                    Bxq�]�  �          @ҏ\@s33�{�@hQ�Bz�C��@s33��Q�?uA33C��                                    Bxq�l�  �          @�(�@�{���@  A���C�7
@�{�333?c�
@�  C�ff                                    Bxq�{F  �          @�(�@��
�
�H@�A��HC�ٚ@��
�<��?
=@�z�C���                                    Bxq���  �          @�z�@�{��ff@���B �HC���@�{�K�@B�\A���C��q                                    Bxq���  �          @�(�@�{���
@X��A���C�aH@�{�9��@
�HA�G�C��q                                    Bxq��8  �          @�
=@��R�Vff?��A���C�@ @��R�u�L�Ϳٙ�C�q�                                    Bxq���  �          @ָR@���Q�?�Q�A�=qC�+�@���s�
�����C�0�                                    Bxq�Ą  �          @�\)@�\)�5�@   A��HC��)@�\)�]p�>.{?�Q�C�p�                                    Bxq��*  �          @�
=@����(Q�@G�A�\)C���@����S�
>�z�@   C�
                                    Bxq���  �          @��@�\)�<��@��A�p�C���@�\)�l��>��
@0��C��R                                    Bxq��v  �          @�33@�33�=p�@p�A�\)C��=@�33�tz�>��H@�  C�AH                                    Bxq��  x          @���@�z��E@z�A��C�3@�z��w
=>���@%�C�4{                                    Bxr �  �          @�33@��QG�@
=A���C���@�����>k�@�\C�\                                    Bxr h  �          @ָR@�z��
=@
=A��HC�T{@�z��#�
?aG�@��C��3                                    Bxr +  T          @ٙ�@�G���G�?��
A
=C�� @�G�����=L��>�C�4{                                    Bxr 9�  �          @ٙ�@ə�����?�p�AlQ�C���@ə��p�>�@��\C�U�                                    Bxr HZ  T          @��@�����@�
A�G�C�AH@��0  ?0��@�33C�H                                    Bxr W   �          @�Q�@ə��޸R?�(�Al(�C�B�@ə����?�\@��
C���                                    Bxr e�  T          @�Q�@���	��@�
A�{C�<)@���:�H?z�@�{C�"�                                    Bxr tL  �          @�  @�{�7
=@p�A�G�C���@�{�fff>���@1�C���                                    Bxr ��  �          @أ�@����*�H@{A��C���@����\��>�G�@mp�C��q                                    Bxr ��  �          @�\)@�ff�ff@�A�{C�4{@�ff�Fff>��H@��C�:�                                    Bxr �>  "          @�Q�@����?�{A�C��@���-p�>��@���C�!H                                    Bxr ��  �          @�G�@Å�  ?�A|��C��@Å�8��>�{@:�HC�Y�                                    Bxr ��  �          @���@�  �'
=?�33Aa�C�<)@�  �E���
���C�b�                                    Bxr �0  �          @�  @��R�<��@33A�C�Q�@��R�fff>\)?��HC��f                                    Bxr ��  �          @��@��H�&ff?�  AK�
C�j=@��H�@  �\)��
=C��H                                    Bxr �|  �          @�=q@�=q�(�?#�
@���C�u�@�=q����E���Q�C��q                                    Bxr �"  �          @�Q�@�{�;�?��
A/
=C��)@�{�J=q�   ��  C��q                                    Bxr�  �          @�  @�Q��I��?��HAH(�C��=@�Q��\(�����\)C���                                    Bxrn  T          @�{@���Q�?�Ah��C��q@���j=q��33�B�\C�O\                                    Bxr$  �          @���@�p��J�H?�Axz�C�h�@�p��h�þ8Q����C��f                                    Bxr2�  T          @׮@�
=��?�(�AJffC��{@�
=�%�=���?Y��C���                                    BxrA`  �          @�\)@���\)?���AW
=C��@���.�R=�G�?xQ�C��                                    BxrP  T          @�
=@���p�?��
Av=qC�ٚ@���A�>8Q�?\C���                                    Bxr^�  �          @��@��H�!�?�z�Aa��C���@��H�AG�=#�
>���C���                                    BxrmR  &          @�  @�\)�5�?���A4z�C�W
@�\)�E����`  C�Q�                                    Bxr{�  b          @ٙ�@���N{?�z�A�C��R@���U�B�\��z�C�&f                                    Bxr��  �          @���@�Q��6ff@G�A��
C��3@�Q��_\)>.{?�
=C�`                                     Bxr�D  �          @�\)@�  �HQ�?��AS�C���@�  �]p��Ǯ�S�
C�xR                                    Bxr��  T          @�\)@�=q�Mp�@{A���C�q�@�=q��G�>�33@A�C�g�                                    Bxr��  �          @�@��\�h��@p  B	�HC�!H@��\���\?���A&�HC�\)                                    Bxr�6  �          @�@�=q�\��@k�Bz�C��{@�=q��z�?��RA,  C���                                    Bxr��  �          @�  @�33��H@3�
A�(�C�L�@�33�c�
?���A{C�Ǯ                                    Bxr�  �          @�=q@Ӆ���?ǮATQ�C�b�@Ӆ����?}p�A�C�7
                                    Bxr�(  �          @ۅ@�{�Q�?��\A+�C��H@�{����?!G�@�G�C�,�                                    Bxr��  �          @ڏ\@��(��?���A"ffC�.@�����?(��@���C��                                    Bxrt  �          @�Q�@��.{?��
Az�C�J=@��0��?G�@��
C��                                    Bxr  T          @أ�@�=q����?ǮAU�C���@�=q��\)?��A33C�*=                                    Bxr+�  �          @أ�@��H��z�@   A��C�Ф@��H�33?��A��C�                                    Bxr:f  �          @�G�@ȣ׿�(�@ffA���C�c�@ȣ���?z�HA��C��                                     BxrI  �          @�Q�@Ӆ�8Q�?�z�A�
C�:�@Ӆ�G�?aG�@�=qC��                                    BxrW�  �          @�G�@�\)?J=q>Ǯ@Tz�@�ff@�\)>�ff?B�\@���@w
=                                    BxrfX  �          @��@�ff?L��?(��@�=q@ٙ�@�ff>��
?}p�A@.{                                    Bxrt�  �          @��@���@  ?��A�RC��=@����  ?\)@��RC��3                                    Bxr��  �          @ڏ\@��Ϳn{?��HA#\)C�f@��Ϳ�
=?�@�33C��=                                    Bxr�J  �          @�G�@�p��s33?O\)@��HC��@�p����R>.{?�z�C��q                                    Bxr��  �          @��H@�p����?��A1��C�p�@�p����H?G�@�33C���                                    Bxr��  �          @��
@�녾�=q>.{?��HC��q@�녾��
    <#�
C��f                                    Bxr�<  �          @���@��>�녿fff����@\(�@��?Tz�
=q��  @�ff                                    Bxr��  �          @��@��
����R���C���@��
>u�
=���
?�(�                                    Bxrۈ  �          @�@����p���Q�5C�t{@����=q��=q���C���                                    Bxr�.  �          @�p�@�=q�u>�@uC��@�=q��ff�����
C��)                                    Bxr��  �          @�@ָR��33?�ffA�C�  @ָR�޸R>�?�{C��R                                    Bxrz  �          @�ff@У׿�\)?��
Am�C�
=@У��z�?!G�@�
=C�33                                    Bxr   �          @�
=@�ff�У�@�
A�\)C���@�ff��R?\(�@��C�~�                                    Bxr$�  
�          @�
=@˅��  @{A���C�H�@˅�+�?n{@��C��
                                    Bxr3l  �          @�p�@���ff@!G�A�33C��@��7�?�z�A=qC���                                    BxrB  �          @��@��Ϳ޸R?��A|��C�h�@����{?&ff@�(�C�o\                                    BxrP�  �          @޸R@�
=��33?�Q�Aap�C��3@�
=� ��>���@U�C�ff                                    Bxr_^  �          @�{@�
=�˅?�A�z�C��@�
=�Q�?B�\@���C��                                    Bxrn  �          @�p�@�(�����?޸RAk�
C��@�(��{>��H@��HC�ff                                    Bxr|�  �          @�Q�@����z�?�
=A^�RC��@�����?J=q@ϮC�)                                    Bxr�P  �          @��
@�z��@
=A�  C��R@�z��9��?&ff@�
=C�^�                                    Bxr��  �          @�
=@�녿���@��A��C�h�@���5�?O\)@���C���                                    Bxr��  �          @߮@�(���?�(�A���C���@�(��3�
?
=q@�C�)                                    Bxr�B  �          @�{@��
���H?�Q�A��C�n@��
�,��?z�@�Q�C��                                    Bxr��  �          @�{@ə���@�A�G�C��@ə��5�?(��@�{C���                                    BxrԎ  �          @�p�@ə�����@��A���C���@ə��.{?aG�@陚C�T{                                    Bxr�4  �          @�ff@θR�Ǯ@33A�
=C�9�@θR�=q?c�
@�33C��H                                    Bxr��  �          @�ff@��H���@33A���C���@��H�'�?��A�
C�                                    Bxr �  �          @�\)@�(��	��@!�A�(�C�Q�@�(��K�?xQ�A z�C�N                                    Bxr&  �          @�@��R���@?\)A���C�:�@��R�L��?�G�AI�C��                                    Bxr�  �          @�{@ə���
=@
=A�33C��@ə��+�?��A��C�y�                                    Bxr,r  �          @���@�z�Tz�@��A���C�Q�@�z�� ��?�\)AY�C�@                                     Bxr;  �          @��@���z�?�z�A�(�C�@���?8Q�@�{C���                                    BxrI�  �          @�ff@љ�����?��HAc�C�G�@љ��\)?��@�{C��3                                    BxrXd  �          @޸R@�G���?�z�A:ffC���@�G���>#�
?��
C��                                    Bxrg
  �          @޸R@�33���
?��A-��C�y�@�33���>��?�  C�Ǯ                                    Bxru�  �          @�@�  ��?�G�AI�C�q@�  �
=>�z�@
=C��                                    Bxr�V  �          @���@����?˅AUp�C���@����>��
@,��C���                                    Bxr��  /          @��H@��H�ff?�(�AF=qC���@��H�#�
=�G�?k�C�H                                    Bxr��  �          @��H@�����?��HAg�
C��@����-p�>��R@%�C�W
                                    Bxr�H  �          @�(�@˅���?�  AIG�C���@˅�'
==�G�?uC��
                                    Bxr��  T          @ۅ@�G��ff?�  Al  C���@�G��-p�>�33@:=qC�\)                                    Bxr͔  T          @�@�Q��p�?���A0  C��@�Q��Q�=u>�C��R                                    Bxr�:  T          @�@Ϯ����?\AK\)C��@Ϯ�Q�>�z�@=qC���                                    Bxr��  y          @�(�@�(�� ��?��AO�
C�Ff@�(��!G�>k�?�33C�9�                                    Bxr��  a          @�z�@�z��?��RAHQ�C��@�z��$z�>\)?��C��                                    Bxr,  �          @��
@���(�?��A\Q�C�o\@���.{>W
=?޸RC�Q�                                    Bxr�  
M          @�(�@ȣ����?�Q�Ad  C�)@ȣ��3�
>aG�?�C���                                    Bxr%x  �          @ۅ@�  ��?�Au�C�b�@�  �3�
>�Q�@A�C���                                    Bxr4  
�          @��
@�=q�Q�@	��A�  C�K�@�=q�J=q?
=q@���C�>�                                    BxrB�  	�          @��
@�����@{A��RC��@���N�R?^�R@��C�Ф                                    BxrQj  
�          @ۅ@�
=�\)@8Q�A�z�C�>�@�
=�h��?���A�
C���                                    Bxr`  �          @ڏ\@�(��5@C33AԸRC��@�(�����?�ffAffC���                                    Bxrn�  �          @ۅ@����1G�@<��Ȁ\C���@����z�H?��\Az�C�P�                                    Bxr}\  �          @ۅ@����<��@Mp�A��
C�b�@������R?���A��C��3                                    Bxr�  �          @�(�@��=q@^�RA�
=C��@��y��?�33A^ffC�*=                                    Bxr��  "          @�(�@����{@7�A�G�C�E@���G
=?�Q�ABffC�!H                                    Bxr�N  �          @�z�@�Q��\@2�\A��C��=@�Q��@A��C�s3                                    Bxr��  �          @�(�@��
��@0  A�33C�t{@��
�*�H?���AS
=C�33                                    Bxrƚ  �          @���@�G��
=q@�RA�\)C�'�@�G��I��?s33@��C�5�                                    